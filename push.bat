@echo off
chcp 65001 >nul
echo กำลังเตรียมข้อมูลเพื่อ Push ขึ้น GitHub...
git status
echo.
set /p msg="ใส่ข้อความ Commit: "
if "%msg%"=="" set msg="Update coloring book app UI/UX"

git add .
git commit -m "%msg%

Co-Authored-By: Claude Code <noreply@anthropic.com>"
git push origin kilo/clear-shell-5yt

echo.
echo ========================================
echo Push เสร็จเรียบร้อยแล้วครับ!
echo ========================================
pause
