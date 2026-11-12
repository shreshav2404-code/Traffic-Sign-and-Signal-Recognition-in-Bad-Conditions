# Open a Word document, update its contents and page fields, save it, export a PDF next to it, and print the page count.
# Usage: powershell -ExecutionPolicy Bypass -File to_pdf.ps1 <path-to-docx>
param([string]$Path)
$full = (Resolve-Path $Path).Path
$pdf = [System.IO.Path]::ChangeExtension($full, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full, $false, $false)
    # Compact contents styles (TOC 1 and TOC 2 are Word's built-in styles -20 and -21), so the contents fits on one page
    foreach ($id in -20, -21) {
        $style = $doc.Styles.Item($id)
        $style.ParagraphFormat.SpaceBefore = 0
        $style.ParagraphFormat.SpaceAfter = 1.5
        $style.ParagraphFormat.LineSpacingRule = 0
        $style.Font.Size = 10
    }
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    # Page numbers in the contents only settle after the first update
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)
    $pages = $doc.ComputeStatistics(2)
    "Pages: $pages"
    "PDF: $pdf"
    $doc.Close($false)
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
