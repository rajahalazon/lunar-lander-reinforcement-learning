"""
build_docx.py
=============

Builds the editable Word version of the report from the same LaTeX source used
for the PDF, so that both carry identical text, numbers, tables and figures.

Pipeline
  1. read equation/figure/table/section numbers and citation numbers from
     report.aux (written by XeLaTeX), and substitute them into a copy of the
     source so that Word shows the same numbering as the PDF;
  2. convert with pandoc (LaTeX -> DOCX, equations become native Word math);
  3. post-process with python-docx: cover page, table of contents, numbered
     display equations, call-out boxes, booktabs-style tables, headers and
     page numbers;
  4. (optional) render with LibreOffice to read the page number of each heading
     and pre-fill the table of contents; Word refreshes it on opening.

Usage:  python build_docx.py   (after compiling report.tex with XeLaTeX)
Requires: pandoc >= 3, python-docx, lxml; LibreOffice for step 4.
"""
from __future__ import annotations

import copy
import re
import shutil
import subprocess
import sys
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor, Emu

HERE = Path(__file__).resolve().parent
TEX = HERE / "report.tex"
AUX = HERE / "report.aux"
BUILD = HERE / "docx_build"
BUILD.mkdir(exist_ok=True)
OUT = HERE.parent / "Final Engineering Report.docx"

NAVY = RGBColor(0x14, 0x28, 0x4B)
ACCENT = RGBColor(0x2A, 0x6C, 0xC0)
CHAR = RGBColor(0x1F, 0x29, 0x33)
SLATE = RGBColor(0x52, 0x60, 0x6D)
BODY_FONT = "Palatino Linotype"
HEAD_FONT = "Calibri"
TEXT_WIDTH_MM = 210 - 2 * 23
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# --------------------------------------------------------------------------
# 1. numbering from the LaTeX aux file
# --------------------------------------------------------------------------
def read_aux():
    labels, cites = {}, {}
    for line in AUX.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", line)
        if m:
            labels[m.group(1)] = m.group(2)
        m = re.match(r"\\bibcite\{([^}]*)\}\{([^}]*)\}", line)
        if m:
            cites[m.group(1)] = int(m.group(2))
    return labels, cites


def cite_text(keys, cites):
    nums = sorted(cites[k.strip()] for k in keys.split(","))
    return "[" + ", ".join(str(n) for n in nums) + "]"


def find_block(s, start_token, open_ch="{", close_ch="}"):
    """Return index just after the balanced group starting at s[start_token]."""
    depth, i = 0, start_token
    while True:
        c = s[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


# --------------------------------------------------------------------------
# 2. pre-process the LaTeX source for pandoc
# --------------------------------------------------------------------------
def preprocess(labels, cites) -> str:
    s = TEX.read_text(encoding="utf-8")
    # drop LaTeX-only preamble and cover; keep \documentclass and macros
    s = re.sub(r"%%BEGIN-LATEX-PREAMBLE.*?%%END-LATEX-PREAMBLE", "", s, flags=re.S)
    s = re.sub(r"%%BEGIN-COVER.*?%%END-COVER", "", s, flags=re.S)
    # front-matter lists and page furniture
    s = re.sub(r"\{\\hypersetup\{linkcolor=charcoal\}\s*\\tableofcontents\}", "", s)
    s = re.sub(r"\{\\hypersetup\{linkcolor=charcoal\}\s*\\listoffigures\s*\\vspace\{4mm\}\s*\\listoftables\}", "", s)
    for pat in [r"\\pagenumbering\{\w+\}", r"\\setcounter\{page\}\{\d+\}", r"\\clearpage",
                r"\\markboth\{[^}]*\}\{[^}]*\}", r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}",
                r"\\setlength\{[^}]*\}\{[^}]*\}", r"\\vspace\{[^}]*\}", r"\\begingroup",
                r"\\endgroup", r"\\renewcommand\{\\section\}\[2\]\{\}"]:
        s = re.sub(pat, "", s)
    # Technical appendices start on a new page in Word too
    s = s.replace("\\section*{Technical Appendices}", "PAGEBREAKMARK\n\n\\section*{Technical Appendices}")

    # objectives list O1..O5 -> bold run-in labels
    def objectives(m):
        body = m.group(1)
        items = [x.strip() for x in body.split("\\item") if x.strip()]
        return "\n\n" + "\n\n".join(f"\\textbf{{O{i}.}} {t}" for i, t in enumerate(items, 1)) + "\n\n"
    s = re.sub(r"\\begin\{enumerate\}\[label=\\textbf\{O\\arabic\*\.\}\](.*?)\\end\{enumerate\}",
               objectives, s, flags=re.S)
    s = re.sub(r"\\begin\{description\}\[[^\]]*\]", r"\\begin{description}", s)

    # figure / table captions carry their number
    def number_floats(env, word):
        nonlocal s
        out, pos = [], 0
        for m in re.finditer(r"\\begin\{%s\}(\[[^\]]*\])?(.*?)\\end\{%s\}" % (env, env), s, flags=re.S):
            body = m.group(2)
            lab = re.search(r"\\label\{([^}]*)\}", body).group(1)
            num = labels[lab]
            c = body.index("\\caption")
            b = c + len("\\caption")
            if body[b] == "[":
                e = body.index("]", b)
                body = body[:b] + body[e + 1:]
            body = body[:b] + "{" + f"\\textbf{{{word} {num}.}} " + body[b + 1:]
            out.append(s[pos:m.start()] + f"\\begin{{{env}}}" + body + f"\\end{{{env}}}")
            pos = m.end()
        out.append(s[pos:])
        s = "".join(out)
    number_floats("figure", "Figure")
    number_floats("table", "Table")

    # numbered display equations -> marker the post-processor turns into a numbered row
    def eq(m):
        body = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", body).group(1)
        body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
        return f"\n\nEQSTART{labels[lab]}EQ\n\\[\n{body}\n\\]\n\n"
    s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq, s, flags=re.S)

    # cross-references and citations
    s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({labels[m.group(1)]})", s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], s)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: cite_text(m.group(1), cites), s)

    # call-out boxes
    s = re.sub(r"\\begin\{keybox\}\{([^}]*)\}", r"\n\nKBSTART\n\n\\textbf{\1}\n\n", s)
    s = s.replace("\\end{keybox}", "\n\nKBEND\n\n")

    # bibliography -> numbered paragraphs
    def bib(m):
        body = m.group(1)
        entries = re.split(r"\\bibitem\{([^}]*)\}", body)[1:]
        paras = []
        for key, text in zip(entries[0::2], entries[1::2]):
            paras.append(f"BIBITEM [{cites[key]}] " + text.strip())
        return "\n\n".join(paras)
    s = re.sub(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", bib, s, flags=re.S)
    s = re.sub(r"\\nolinkurl\{([^}]*)\}", lambda m: "\\texttt{" + m.group(1).replace("_", "\\_") + "}", s)

    # figures: use the 300-dpi PNG exports
    s = re.sub(r"\\includegraphics\[([^\]]*)\]\{(Fig[^}]*)\.pdf\}",
               r"\\includegraphics[\1]{../Figures/\2.png}", s)
    # misc commands pandoc does not need
    s = s.replace("\\centering", "").replace("\\footnotesize", "").replace("\\small", "")
    s = s.replace("\\noindent", "")
    return s


# --------------------------------------------------------------------------
# 3. reference document (styles)
# --------------------------------------------------------------------------
def set_font(style, name, size=None, color=None, bold=None, italic=None):
    f = style.font
    f.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]
    if size:
        f.size = Pt(size)
    if color is not None:
        f.color.rgb = color
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic


def make_reference(path: Path):
    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    d = docx.Document(str(path))

    class _Styles:
        def __init__(self, styles):
            self._m = {x.name: x for x in styles}
        def __getitem__(self, k):
            return self._m[k]
        def __iter__(self):
            return iter(self._m.values())
    st = _Styles(d.styles)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Definition", "Bibliography",
                 "Block Text", "Abstract"):
        if name in [x.name for x in st]:
            set_font(st[name], BODY_FONT, 10.5, CHAR)
    for name in ("Body Text", "First Paragraph"):
        pf = st[name].paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(6), 1.12
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = st["Compact"].paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(2.5), 1.1
    sizes = {"Heading 1": 15, "Heading 2": 12.5, "Heading 3": 11}
    for h, sz in sizes.items():
        set_font(st[h], HEAD_FONT, sz, NAVY, bold=True, italic=False)
        pf = st[h].paragraph_format
        pf.space_before = Pt(18 if h == "Heading 1" else 12)
        pf.space_after = Pt(6 if h == "Heading 1" else 4)
        pf.keep_with_next = True
    for name in ("Image Caption", "Table Caption", "Caption"):
        set_font(st[name], BODY_FONT, 9, CHAR, italic=False)
        pf = st[name].paragraph_format
        pf.space_before, pf.space_after = Pt(3), Pt(10)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    st["Table Caption"].paragraph_format.space_after = Pt(4)
    st["Table Caption"].paragraph_format.keep_with_next = True
    st["Definition Term"].font.bold = True
    set_font(st["Definition Term"], BODY_FONT, 10.5, CHAR, bold=True)
    st["Hyperlink"].font.color.rgb = NAVY
    st["Verbatim Char"].font.name = "Consolas"
    st["Verbatim Char"].font.size = Pt(9.5)
    sec = d.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(23)
    sec.top_margin = sec.bottom_margin = Mm(25)
    sec.header_distance = sec.footer_distance = Mm(12)
    d.save(str(path))


# --------------------------------------------------------------------------
# 4. post-processing helpers
# --------------------------------------------------------------------------
def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def para_text(p_el):
    return "".join(t.text or "" for t in p_el.iter(qn("w:t")))


def set_cell_borders(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = el("w:tcBorders")
        tcPr.append(b)
    for edge, spec in edges.items():
        e = el(f"w:{edge}", **{"w:val": spec[0], "w:sz": spec[1], "w:space": 0, "w:color": spec[2]})
        old = b.find(qn(f"w:{edge}"))
        if old is not None:
            b.remove(old)
        b.append(e)


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = el("w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": fill})
    tcPr.append(sh)


def no_borders(table):
    tblPr = table._tbl.tblPr
    b = el("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b.append(el(f"w:{edge}", **{"w:val": "nil"}))
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    tblPr.append(b)


def set_table_width(table, mm, widths_mm=None):
    tblPr = table._tbl.tblPr
    for t in tblPr.findall(qn("w:tblW")):
        tblPr.remove(t)
    tblPr.append(el("w:tblW", **{"w:w": int(mm / 25.4 * 1440), "w:type": "dxa"}))
    lay = el("w:tblLayout", **{"w:type": "fixed"})
    tblPr.append(lay)
    if widths_mm:
        grid = table._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths_mm):
            gc.set(qn("w:w"), str(int(w / 25.4 * 1440)))
        for row in table.rows:
            for c, w in zip(row.cells, widths_mm):
                c.width = Mm(w)


def add_run(p, text, font=HEAD_FONT, size=10, color=CHAR, bold=False, italic=False):
    r = p.add_run(text)
    r.font.name = font
    r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), font)
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    return r


def add_field(p, instr, placeholder="", size=None, font=None, color=None):
    def run_with(child):
        r = el("w:r")
        if size or font or color:
            rpr = el("w:rPr")
            if font:
                rpr.append(el("w:rFonts", **{"w:ascii": font, "w:hAnsi": font}))
            if color:
                rpr.append(el("w:color", **{"w:val": color}))
            if size:
                rpr.append(el("w:sz", **{"w:val": int(size * 2)}))
            r.append(rpr)
        r.append(child)
        return r
    p._p.append(run_with(el("w:fldChar", **{"w:fldCharType": "begin", "w:dirty": "true"})))
    it = el("w:instrText")
    it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    it.text = f" {instr} "
    p._p.append(run_with(it))
    p._p.append(run_with(el("w:fldChar", **{"w:fldCharType": "separate"})))
    t = el("w:t")
    t.text = placeholder
    p._p.append(run_with(t))
    p._p.append(run_with(el("w:fldChar", **{"w:fldCharType": "end"})))


def page_break_paragraph(doc_body_el, before_el):
    p = el("w:p")
    r = el("w:r")
    r.append(el("w:br", **{"w:type": "page"}))
    p.append(r)
    before_el.addprevious(p)
    return p



# --------------------------------------------------------------------------
# OOXML schema order: Word rejects property elements that are out of order or
# duplicated, so every property container is normalized before saving.
# --------------------------------------------------------------------------
_ORDER = {
    "rPr": "rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof "
           "snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd "
           "fitText vertAlign rtl cs em lang eastAsianLayout specVanish oMath",
    "pPr": "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd "
           "tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi "
           "adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc "
           "textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
    "tblPr": "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc "
             "tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription",
    "tcPr": "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark",
    "trPr": "cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden",
    "tcBorders": "top start left bottom end right insideH insideV tl2br tr2bl",
    "tblBorders": "top start left bottom end right insideH insideV",
    "pBdr": "top left bottom right between bar",
    "tcMar": "top start left bottom end right",
    "tblCellMar": "top start left bottom end right",
}
_ORDER = {k: v.split() for k, v in _ORDER.items()}
_WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def normalize_xml(root):
    for name, order in _ORDER.items():
        for cont in root.iter(_WNS + name):
            kids = [k for k in cont if isinstance(k.tag, str)]
            last = {}
            for k in kids:
                if k.tag.startswith(_WNS):
                    last[k.tag] = k
            keep = [k for k in kids if not k.tag.startswith(_WNS) or last.get(k.tag) is k]
            for k in kids:
                cont.remove(k)
            def key(k):
                t = k.tag.replace(_WNS, "")
                return order.index(t) if t in order else len(order)
            for k in sorted(keep, key=key):
                cont.append(k)



_MORDER = {
    "rPr": "lit nor scr sty brk aln", "dPr": "begChr sepChr endChr grow shp ctrlPr",
    "mcPr": "count mcJc", "mPr": "baseJc plcHide rSpRule cGpRule rSp cSp cGp mcs ctrlPr",
    "naryPr": "chr limLoc grow subHide supHide ctrlPr", "fPr": "type ctrlPr", "accPr": "chr ctrlPr",
    "groupChrPr": "chr pos vertJc ctrlPr", "eqArrPr": "baseJc maxDist objDist rSpRule rSp ctrlPr",
    "barPr": "pos ctrlPr", "radPr": "degHide ctrlPr", "boxPr": "opEmu noBreak diff brk aln ctrlPr",
    "borderBoxPr": "hideTop hideBot hideLeft hideRight strikeH strikeV strikeBLTR strikeTLBR ctrlPr",
}
_MNS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
_STYLE_ORDER = ("name aliases basedOn next link autoRedefine hidden uiPriority semiHidden unhideWhenUsed "
                "qFormat locked personal personalCompose personalReply rsid pPr rPr tblPr trPr tcPr tblStylePr").split()
_SETTINGS_ORDER = ("writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries "
    "displayBackgroundShape printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts "
    "embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins alignBordersAndEdges bordersDoNotSurroundHeader "
    "bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle proofState "
    "formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge "
    "revisionView trackRevisions doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride "
    "styleLockTheme styleLockQFSet defaultTabStop autoHyphenation consecutiveHyphenLimit hyphenationZone "
    "doNotHyphenateCaps showEnvelope summaryLength clickAndTypeStyle defaultTableStyle evenAndOddHeaders "
    "bookFoldRevPrinting bookFoldPrinting bookFoldPrintingSheets drawingGridHorizontalSpacing "
    "drawingGridVerticalSpacing displayHorizontalDrawingGridEvery displayVerticalDrawingGridEvery "
    "doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin drawingGridVerticalOrigin doNotShadeFormData "
    "noPunctuationKerning characterSpacingControl printTwoOnOne strictFirstAndLastChars noLineBreaksAfter "
    "noLineBreaksBefore savePreviewPicture doNotValidateAgainstSchema saveInvalidXml ignoreMixedContent "
    "alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly useXSLTWhenSaving saveThroughXslt "
    "showXMLTags alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr endnotePr compat docVars "
    "rsids mathPr attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats "
    "doNotAutoCompressPictures forceUpgrade captions readModeInkLockDown smartTagType schemaLibrary "
    "shapeDefaults doNotEmbedSmartTags decimalSymbol listSeparator").split()


def _sort_children(cont, order, ns):
    kids = [k for k in cont if isinstance(k.tag, str)]
    for k in kids:
        cont.remove(k)
    key = lambda k: order.index(k.tag.replace(ns, "")) if k.tag.startswith(ns) and k.tag.replace(ns, "") in order else len(order)
    for k in sorted(kids, key=key):
        cont.append(k)


def fix_package(path: Path):
    """Final schema clean-up of parts produced by pandoc and python-docx."""
    import zipfile
    from lxml import etree
    tmp = path.with_suffix(".tmp")
    zin = zipfile.ZipFile(path)
    zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename in ("word/document.xml", "word/styles.xml", "word/settings.xml",
                             "word/numbering.xml") or item.filename.startswith(("word/header", "word/footer")):
            root = etree.fromstring(data)
            normalize_xml(root)
            for name, order in _MORDER.items():
                for c in root.iter(_MNS + name):
                    _sort_children(c, order.split(), _MNS)
            for st in root.iter(_WNS + "style"):
                _sort_children(st, _STYLE_ORDER, _WNS)
            if item.filename == "word/settings.xml":
                kids = [k for k in root if isinstance(k.tag, str)]
                for k in kids:
                    root.remove(k)
                loc = lambda k: k.tag.split("}")[1]
                for k in sorted(kids, key=lambda k: _SETTINGS_ORDER.index(loc(k)) if loc(k) in _SETTINGS_ORDER
                                else len(_SETTINGS_ORDER)):
                    root.append(k)
            for e in root.iter():  # stray text (e.g. a '>' in pandoc's default styles)
                if e.tag in (_WNS + "rPr", _WNS + "pPr"):
                    e.text = None
                    for k in e:
                        if k.tail and k.tail.strip():
                            k.tail = None
            for mr in root.iter(_MNS + "rPr"):  # m:nor and m:sty are mutually exclusive
                if mr.find(_MNS + "nor") is not None:
                    for st_ in mr.findall(_MNS + "sty"):
                        mr.remove(st_)
            for n in root.iter(_WNS + "nsid"):
                v = n.get(_WNS + "val")
                n.set(_WNS + "val", f"{int(v, 16) & 0xFFFFFFFF:08X}")
            for pm in root.iter(_WNS + "pgMar"):
                for a, dv in (("header", "680"), ("footer", "680"), ("gutter", "0")):
                    if pm.get(_WNS + a) is None:
                        pm.set(_WNS + a, dv)
            data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        zout.writestr(item, data)
    zout.close()
    zin.close()
    tmp.replace(path)

# --------------------------------------------------------------------------
# 5. post-processing
# --------------------------------------------------------------------------
def postprocess(src: Path, dst: Path, toc_pages: dict | None = None,
                lof: list | None = None, lot: list | None = None):
    d = docx.Document(str(src))
    body = d.element.body
    W = TEXT_WIDTH_MM

    # ---- numbered display equations: 3-cell borderless table -------------
    for p in list(body.iter(qn("w:p"))):
        txt = para_text(p)
        m = re.fullmatch(r"\s*EQSTART([\w.]+)EQ\s*", txt)
        if not m:
            continue
        num = m.group(1)
        nxt = p.getnext()
        omp = None
        if nxt is not None and nxt.tag == qn("w:p"):
            omp = nxt.find(qn("m:oMathPara"))
        if omp is None:  # math may be in the same paragraph
            omp = p.find(qn("m:oMathPara"))
            nxt = p
        tbl = d.add_table(rows=1, cols=3)
        no_borders(tbl)
        set_table_width(tbl, W, [W * 0.08, W * 0.84, W * 0.08])
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        c0, c1, c2 = tbl.rows[0].cells
        mp = c1.paragraphs[0]._p
        mp.append(copy.deepcopy(omp))
        c1.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for c in (c0, c1, c2):
            pf = c.paragraphs[0].paragraph_format
            pf.space_before, pf.space_after = Pt(3), Pt(3)
        c1.vertical_alignment = c2.vertical_alignment = 1
        c2.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        add_run(c2.paragraphs[0], f"({num})", font=BODY_FONT, size=10.5)
        p.addprevious(tbl._tbl)
        if nxt is not p:
            body.remove(nxt)
        body.remove(p)

    # ---- call-out boxes --------------------------------------------------
    while True:
        start = next((p for p in body.iter(qn("w:p")) if para_text(p).strip() == "KBSTART"), None)
        if start is None:
            break
        items, cur = [], start.getnext()
        while para_text(cur).strip() != "KBEND":
            items.append(cur)
            cur = cur.getnext()
        end = cur
        tbl = d.add_table(rows=1, cols=1)
        no_borders(tbl)
        set_table_width(tbl, W, [W])
        cell = tbl.rows[0].cells[0]
        shade_cell(cell, "F3F6FA")
        set_cell_borders(cell, left=("single", 18, "14284B"))
        cell._tc.remove(cell.paragraphs[0]._p)
        for i, it in enumerate(items):
            cell._tc.append(it)
            for r in it.iter(qn("w:r")):
                rpr = r.find(qn("w:rPr"))
                if rpr is None:
                    rpr = el("w:rPr")
                    r.insert(0, rpr)
                rpr.append(el("w:sz", **{"w:val": 19}))
                if i == 0:
                    rpr.append(el("w:color", **{"w:val": "14284B"}))
                    rpr.insert(0, el("w:rFonts", **{"w:ascii": HEAD_FONT, "w:hAnsi": HEAD_FONT}))
        mar = el("w:tcMar")
        for edge, v in (("top", 100), ("bottom", 100), ("left", 170), ("right", 170)):
            mar.append(el(f"w:{edge}", **{"w:w": v, "w:type": "dxa"}))
        cell._tc.get_or_add_tcPr().append(mar)
        start.addprevious(tbl._tbl)
        spacer = el("w:p")
        tbl._tbl.addnext(spacer)
        body.remove(start)
        body.remove(end)

    # ---- page-break marker and bibliography styling -----------------------
    for p in list(body.iter(qn("w:p"))):
        t = para_text(p).strip()
        if t == "PAGEBREAKMARK":
            for r in list(p.iter(qn("w:r"))):
                p.remove(r) if r.getparent() is p else None
            r = el("w:r")
            r.append(el("w:br", **{"w:type": "page"}))
            p.append(r)
        elif t.startswith("BIBITEM"):
            first_t = next(p.iter(qn("w:t")))
            first_t.text = first_t.text.replace("BIBITEM ", "", 1).replace("BIBITEM", "", 1)
            pp = docx.text.paragraph.Paragraph(p, d._body)
            pf = pp.paragraph_format
            pf.left_indent, pf.first_line_indent = Mm(9), Mm(-9)
            pf.space_after, pf.alignment = Pt(4), WD_ALIGN_PARAGRAPH.LEFT
            for r in pp.runs:
                r.font.size = Pt(9.5)

    # ---- data tables: booktabs-style rules, compact font -----------------
    for tbl in d.tables:
        tblPr = tbl._tbl.tblPr
        style = tblPr.find(qn("w:tblStyle"))
        if style is None or style.get(qn("w:val")) != "Table":
            continue
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for t in tblPr.findall(qn("w:tblW")):
            tblPr.remove(t)
        tblPr.append(el("w:tblW", **{"w:w": 5000, "w:type": "pct"}))
        for t in tblPr.findall(qn("w:tblLayout")):
            tblPr.remove(t)
        tblPr.append(el("w:tblLayout", **{"w:type": "fixed"}))
        ncols = len(tbl.columns)
        def clen(c):
            return len("".join(t.text or "" for t in c._tc.iter(qn("w:t"), qn("m:t"))))
        lens = []
        for ci, col in enumerate(tbl.columns):
            body_len = max([clen(c) for c in col.cells[1:]] or [1])
            head_len = max(clen(col.cells[0]) * 1.0, 1)
            lens.append(min(max(body_len, head_len, 5) + 1.5, 55))
        tot = sum(lens)
        mar = el("w:tblCellMar")
        for edge, v in (("top", 0), ("left", 57), ("bottom", 0), ("right", 57)):
            mar.append(el(f"w:{edge}", **{"w:w": v, "w:type": "dxa"}))
        for t in tblPr.findall(qn("w:tblCellMar")):
            tblPr.remove(t)
        tblPr.append(mar)
        widths = [TEXT_WIDTH_MM * L / tot for L in lens]
        grid = tbl._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w / 25.4 * 1440)))
        for row in tbl.rows:
            for c, w in zip(row.cells, widths):
                c.width = Mm(w)
        nrows = len(tbl.rows)
        for ri, row in enumerate(tbl.rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(el("w:cantSplit"))
            for cell in row.cells:
                edges = {}
                if ri == 0:
                    edges["top"] = ("single", 10, "1F2933")
                    edges["bottom"] = ("single", 5, "1F2933")
                if ri == nrows - 1:
                    edges["bottom"] = ("single", 10, "1F2933")
                if edges:
                    set_cell_borders(cell, **edges)
                for p in cell.paragraphs:
                    pf = p.paragraph_format
                    pf.space_before, pf.space_after = Pt(1.5), Pt(1.5)
                    pf.line_spacing = 1.0
                    for r in p.runs:
                        r.font.size = Pt(8)
                        if ri == 0:
                            r.font.bold = False
                            r.font.color.rgb = NAVY
            if ri == 0:
                trPr.append(el("w:tblHeader"))

    # ---- centre figures ---------------------------------------------------
    for p in d.paragraphs:
        if p.style.name in ("Captioned Figure", "Figure") or p._p.find(".//" + qn("w:drawing")) is not None:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True

    # ---- headings: no page break issues, colour section numbers ----------
    for p in d.paragraphs:
        if p.style.name.startswith("Heading"):
            for r in p.runs:
                r.font.name = HEAD_FONT

    # ---- front matter: cover, contents, lists -----------------------------
    first = body[0]
    intro = next(p for p in d.paragraphs if p.style.name == "Heading 1" and "Introduction" in p.text)
    build_cover(d, first)
    build_lists(d, intro._p, toc_pages, lof, lot)

    # ---- section, header, footer -----------------------------------------
    sec = d.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = Mm(23)
    sec.top_margin = sec.bottom_margin = Mm(25)
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    hp.text = ""
    add_run(hp, "REINFORCEMENT LEARNING FOR AUTONOMOUS LUNAR LANDING", size=7.5, color=SLATE)
    hp.paragraph_format.tab_stops.add_tab_stop(Mm(W), WD_TAB_ALIGNMENT.RIGHT)
    add_run(hp, "\tTechnical report", size=7.5, color=SLATE)
    pPr = hp._p.get_or_add_pPr()
    bdr = el("w:pBdr")
    bdr.append(el("w:bottom", **{"w:val": "single", "w:sz": 4, "w:space": 4, "w:color": "9AA5B1"}))
    pPr.append(bdr)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(fp, "PAGE", "1", size=9, font=HEAD_FONT, color="52606D")

    # ---- front matter in its own section, numbered i, ii, ...; body restarts at 1
    final_sp = body.find(qn("w:sectPr"))
    fm_sp = copy.deepcopy(final_sp)
    for t in fm_sp.findall(qn("w:pgNumType")):
        fm_sp.remove(t)
    fm_sp.append(el("w:pgNumType", **{"w:fmt": "lowerRoman", "w:start": 1}))
    for t in final_sp.findall(qn("w:titlePg")):
        final_sp.remove(t)
    for t in final_sp.findall(qn("w:pgNumType")):
        final_sp.remove(t)
    final_sp.append(el("w:pgNumType", **{"w:fmt": "decimal", "w:start": 1}))
    # order matters inside sectPr: pgNumType before titlePg etc.; re-sort children
    _order = ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz", "pgMar",
              "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols", "formProt", "vAlign",
              "noEndnote", "titlePg", "textDirection", "bidi", "rtlGutter", "docGrid", "printerSettings"]
    for sp in (fm_sp, final_sp):
        kids = list(sp)
        kids.sort(key=lambda k: _order.index(k.tag.split("}")[1]) if k.tag.split("}")[1] in _order else 99)
        for k in kids:
            sp.remove(k)
        for k in kids:
            sp.append(k)
    intro_el = intro._p
    prev = intro_el.getprevious()
    if prev is not None and prev.find(".//" + qn("w:br")) is not None and not para_text(prev).strip():
        body.remove(prev)
    brk = el("w:p")
    ppr = el("w:pPr")
    ppr.append(fm_sp)
    brk.append(ppr)
    intro_el.addprevious(brk)

    # ---- ask Word to refresh fields (TOC page numbers) on opening ---------
    settings = d.settings.element
    uf = settings.find(qn("w:updateFields"))
    if uf is None:
        settings.append(el("w:updateFields", **{"w:val": "true"}))

    normalize_xml(d.element)
    for sec_ in d.sections:
        for part in (sec_.header, sec_.footer):
            normalize_xml(part._element)
    cp = d.core_properties
    cp.title = "Reinforcement Learning for Autonomous Lunar Landing: Reward Shaping, Policy Optimization, and Reward Hacking"
    cp.author = "Raja Halazon"
    cp.subject = "Engineering research report: PPO reward-shaping study in Gymnasium LunarLander-v3"
    cp.keywords = "reinforcement learning; PPO; reward shaping; reward hacking; LunarLander"
    d.save(str(dst))
    fix_package(dst)


def build_cover(d, first_el):
    W = TEXT_WIDTH_MM
    els = []

    def P(space_before=0, space_after=0):
        p = d.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        els.append(p._p)
        return p

    p = P(0, 2)
    p.paragraph_format.tab_stops.add_tab_stop(Mm(W), WD_TAB_ALIGNMENT.RIGHT)
    add_run(p, "INDEPENDENT ENGINEERING PROJECT", size=8.5, color=NAVY, bold=True)
    add_run(p, "\tTECHNICAL REPORT", size=8.5, color=NAVY)
    pPr = p._p.get_or_add_pPr()
    b = el("w:pBdr")
    b.append(el("w:bottom", **{"w:val": "single", "w:sz": 12, "w:space": 6, "w:color": "14284B"}))
    pPr.append(b)
    p = P(150, 0)
    add_run(p, "Reinforcement Learning for", size=28, color=NAVY, bold=True)
    p = P(0, 14)
    add_run(p, "Autonomous Lunar Landing", size=28, color=NAVY, bold=True)
    p = P(0, 18)
    add_run(p, "Reward Shaping, Policy Optimization, and Reward Hacking", font=BODY_FONT, size=16,
            color=CHAR, italic=True)
    p = P(0, 18)
    pPr = p._p.get_or_add_pPr()
    b = el("w:pBdr")
    b.append(el("w:bottom", **{"w:val": "single", "w:sz": 18, "w:space": 1, "w:color": "2A6CC0"}))
    pPr.append(b)
    ind = el("w:ind", **{"w:right": int((W - 38) / 25.4 * 1440)})
    pPr.append(ind)
    p = P(0, 3)
    add_run(p, "Raja Halazon", size=13, color=CHAR, bold=True)
    p = P(0, 1)
    add_run(p, "Mechanical Engineering, McGill University", size=11, color=SLATE)
    p = P(0, 0)
    add_run(p, "October 2026", size=11, color=SLATE)

    t = d.add_table(rows=2, cols=3)
    no_borders(t)
    set_table_width(t, W, [W / 3] * 3)
    labels = ["ENVIRONMENT", "ALGORITHM", "EVIDENCE BASE"]
    vals = ["Gymnasium LunarLander-v3", "PPO, Stable-Baselines3", "4 agents, 400 evaluation episodes"]
    for i in range(3):
        c = t.rows[0].cells[i]
        add_run(c.paragraphs[0], labels[i], size=8, color=SLATE)
        set_cell_borders(c, top=("single", 4, "9AA5B1"))
        c.paragraphs[0].paragraph_format.space_before = Pt(6)
        c2 = t.rows[1].cells[i]
        add_run(c2.paragraphs[0], vals[i], size=10.5, color=NAVY)
        set_cell_borders(c2, bottom=("single", 4, "9AA5B1"))
        c2.paragraphs[0].paragraph_format.space_after = Pt(6)
    P(190, 0)  # spacer above the information strip
    p = P(6, 0)
    add_run(p, "Report version 2.0. All quantitative results were recomputed from the original evaluation "
               "dataset (final_results.csv). This is a simulation study; no flight hardware or flight "
               "software is described.", size=7.5, color=SLATE)
    r = el("w:r")
    r.append(el("w:br", **{"w:type": "page"}))
    p._p.append(r)
    # order: header, title..., date, spacer, table, note
    order = els[:-2] + [els[-2], t._tbl, els[-1]]
    for e in order:
        first_el.addprevious(e)


def toc_line(d, text, page, level, bold=False):
    p = d.add_paragraph()
    pf = p.paragraph_format
    pf.tab_stops.add_tab_stop(Mm(TEXT_WIDTH_MM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    pf.left_indent = Mm(0 if level == 1 else 8)
    pf.space_before = Pt(4 if level == 1 else 0)
    pf.space_after = Pt(1)
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_run(p, f"{text}\t{page}", font=HEAD_FONT if level == 1 else BODY_FONT,
            size=10 if level == 1 else 9.5, color=NAVY if level == 1 else CHAR, bold=bold)
    return p


def build_lists(d, before_el, toc_pages, lof, lot):
    """Contents, figures and tables as Word fields pre-filled with page numbers."""
    def heading(text):
        p = d.add_paragraph()
        add_run(p, text, size=15, color=NAVY, bold=True)
        p.paragraph_format.space_after = Pt(10)
        return p

    blocks = []
    pb0 = d.add_paragraph()
    pb0.add_run().add_break(WD_BREAK.PAGE)
    blocks.append(pb0._p)
    h = heading("Contents")
    blocks.append(h._p)
    entries = toc_pages or []
    fld_p = d.add_paragraph()
    blocks.append(fld_p._p)
    # field begin in first paragraph, entries as result, end after last entry
    add_field_begin(fld_p, 'TOC \\o "1-2" \\h \\z \\u')
    if not entries:
        entries = [("Update this table (right-click > Update Field).", "", 1)]
    for text, page, level in entries:
        blocks.append(toc_line(d, text, page, level, bold=(level == 1))._p)
    endp = d.add_paragraph()
    add_field_end(endp)
    blocks.append(endp._p)
    pb = d.add_paragraph()
    pb.add_run().add_break(WD_BREAK.PAGE)
    blocks.append(pb._p)
    for title, items, seq in (("List of Figures", lof, "Figure"), ("List of Tables", lot, "Table")):
        blocks.append(heading(title)._p)
        fp = d.add_paragraph()
        blocks.append(fp._p)
        add_field_begin(fp, f'TOC \\h \\z \\c "{seq}"')
        for text, page in (items or []):
            blocks.append(toc_line(d, text, page, 2)._p)
        ep = d.add_paragraph()
        add_field_end(ep)
        blocks.append(ep._p)
    pb = d.add_paragraph()
    pb.add_run().add_break(WD_BREAK.PAGE)
    blocks.append(pb._p)
    for b in blocks:
        before_el.addprevious(b)


def add_field_begin(p, instr):
    r = el("w:r"); r.append(el("w:fldChar", **{"w:fldCharType": "begin", "w:dirty": "true"})); p._p.append(r)
    r = el("w:r"); it = el("w:instrText"); it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    it.text = f" {instr} "; r.append(it); p._p.append(r)
    r = el("w:r"); r.append(el("w:fldChar", **{"w:fldCharType": "separate"})); p._p.append(r)


def add_field_end(p):
    r = el("w:r"); r.append(el("w:fldChar", **{"w:fldCharType": "end"})); p._p.append(r)


# --------------------------------------------------------------------------
# 6. page numbers for the contents, read from a LibreOffice rendering
# --------------------------------------------------------------------------
def render_pdf(docx_path: Path) -> Path:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(BUILD), str(docx_path)],
                   check=True, capture_output=True)
    return BUILD / (docx_path.stem + ".pdf")


def page_texts(pdf: Path):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                           text=True).stdout).group(1))
    return [subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), str(pdf), "-"],
                           capture_output=True, text=True).stdout for i in range(1, n + 1)]


def collect_entries(docx_path: Path, pdf: Path):
    d = docx.Document(str(docx_path))
    pages = page_texts(pdf)
    norm = lambda x: re.sub(r"\s+", " ", x).strip()
    npages = [norm(p) for p in pages]
    # body page numbering starts at the page that holds "1 Introduction"
    lot_idx = max(i for i, p in enumerate(npages) if "List of Tables" in p)
    start = next(i for i in range(lot_idx + 1, len(npages)) if "Introduction" in npages[i])

    def find(text, frm):
        key = norm(text)[:60]
        for i in range(frm, len(npages)):
            if key in npages[i]:
                return i
        return None

    toc, lof, lot = [], [], []
    cursor = start
    for p in d.paragraphs:
        if p.style.name == "Heading 1" and p.text.strip() == "Abstract":
            toc.append(("Abstract", _roman(next(i for i, q in enumerate(npages) if q.startswith(
                "REINFORCEMENT") and "Abstract" in q[:120]) + 1), 1))
            continue
        sn = p.style.name
        if sn in ("Heading 1", "Heading 2") and p.text.strip():
            i = find(p.text, cursor)
            if i is None:
                continue
            cursor = i
            label = re.sub(r"\s+", "  ", p.text.strip())
            num = str(i - start + 1) if i >= start else _roman(i + 1)
            toc.append((label, num, 1 if sn == "Heading 1" else 2))
        elif sn in ("Image Caption", "Table Caption") and p.text.strip():
            m = re.match(r"(Figure|Table) (\d+)\.\s*(.*)", p.text.strip())
            if not m:
                continue
            i = find(p.text.strip()[:50], cursor)
            if i is None:
                continue
            cursor = min(cursor, i) if sn == "Table Caption" else cursor
            short = SHORT.get((m.group(1), m.group(2)), m.group(3)[:70])
            entry = (f"{m.group(1)} {m.group(2)}.  {short}", str(i - start + 1))
            (lof if m.group(1) == "Figure" else lot).append(entry)
    return toc, lof, lot


def _roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


SHORT: dict = {}


def load_short_captions():
    s = TEX.read_text(encoding="utf-8")
    labels, _ = read_aux()
    for m in re.finditer(r"\\caption\[([^\]]*)\]\{.*?\\label\{([^}]*)\}", s, flags=re.S):
        lab = m.group(2)
        kind = "Figure" if lab.startswith("fig:") else "Table"
        SHORT[(kind, labels[lab])] = m.group(1).replace("--", "–")


def main():
    labels, cites = read_aux()
    load_short_captions()
    src = preprocess(labels, cites)
    (BUILD / "docx_input.tex").write_text(src, encoding="utf-8")
    ref = BUILD / "reference.docx"
    make_reference(ref)
    stage1 = BUILD / "stage1.docx"
    subprocess.run(["pandoc", str(BUILD / "docx_input.tex"), "-f", "latex", "-t", "docx",
                    "--number-sections", f"--reference-doc={ref}", "--resource-path", str(HERE),
                    "-o", str(stage1)], check=True, cwd=str(HERE))
    stage2 = BUILD / "stage2.docx"
    postprocess(stage1, stage2)
    if shutil.which("soffice") and "--no-pages" not in sys.argv:
        pdf = render_pdf(stage2)
        toc, lof, lot = collect_entries(stage2, pdf)
        postprocess(stage1, OUT, toc, lof, lot)
        # second pass: lists may shift pagination by a page; recompute once
        pdf2 = render_pdf(OUT)
        toc2, lof2, lot2 = collect_entries(OUT, pdf2)
        if (toc2, lof2, lot2) != (toc, lof, lot):
            postprocess(stage1, OUT, toc2, lof2, lot2)
    else:
        shutil.copy(stage2, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
