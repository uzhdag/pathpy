
__all__ = ["svg_to_pdf", 'svg_to_png']

def svg_to_pdf(svg_file, output_file):
    # uses svglib to render a SVG to PDF
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    drawing = svg2rlg(svg_file)
    renderPDF.drawToFile(drawing, output_file)


def svg_to_png(svg_file, output_file):
    # uses svglib to render a SVG to PDF
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM

    drawing = svg2rlg(svg_file)
    renderPM.drawToFile(drawing, output_file, fmt='PNG')