# Convert PPTX to PDF using PowerPoint COM automation
param(
    [string]$pptx = "$PSScriptRoot\Financial_QA_Presentation.pptx",
    [string]$pdf  = "$PSScriptRoot\Financial_QA_Presentation.pdf"
)

Write-Host "Converting $pptx -> $pdf"

$powerpoint = New-Object -ComObject PowerPoint.Application
$presentation = $powerpoint.Presentations.Open($pptx, $true, $true, $false)

# 32 = PpSaveAsFileType.ppSaveAsPDF
$presentation.SaveAs($pdf, 32)
$presentation.Close()
$powerpoint.Quit()

[System.Runtime.Interopservices.Marshal]::ReleaseComObject($powerpoint) | Out-Null

Write-Host "Done: $pdf"
