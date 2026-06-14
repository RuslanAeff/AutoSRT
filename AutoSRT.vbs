' AutoSRT - konsolsuz baslatici (CMD penceresi acmaz)
' main.py dosyasini pythonw.exe ile gizli olarak calistirir.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' pythonw.exe'yi PATH uzerinden bul; bulunamazsa python kullan (yine de gizli)
pyw = "pythonw.exe"

sh.CurrentDirectory = scriptDir
' 0 = pencere gizli, False = bitmesini bekleme
sh.Run """" & pyw & """ """ & scriptDir & "\main.py""", 0, False
