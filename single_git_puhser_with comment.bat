@echo off
git status
set /p msg=Added commitment remark:
git add .
git commit -m "%msg%"
git pull
git push
echo Succeed: %msg%
echo --------End!--------
pause