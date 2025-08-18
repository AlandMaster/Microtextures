echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="D:\Program Files\ANSYS Inc\ANSYS Student\v232\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "D:\Program Files\ANSYS Inc\ANSYS Student\v232\fluent\ntbin\win64\tell.exe" Valkyrje 64035 CLEANUP_EXITING
timeout /t 1
"D:\Program Files\ANSYS Inc\ANSYS Student\v232\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="Valkyrje" (%KILL_CMD% 19824) 
if /i "%LOCALHOST%"=="Valkyrje" (%KILL_CMD% 14428) 
if /i "%LOCALHOST%"=="Valkyrje" (%KILL_CMD% 8856) 
if /i "%LOCALHOST%"=="Valkyrje" (%KILL_CMD% 14052) 
if /i "%LOCALHOST%"=="Valkyrje" (%KILL_CMD% 3764) 
if /i "%LOCALHOST%"=="Valkyrje" (%KILL_CMD% 15108)
del "D:\Work\microtex\cleanup-fluent-Valkyrje-3764.bat"
