# Commands

## Copy a folder

cp -r /path/source /path/to

-r: 表示recursively，在移动文件夹时需要使用

scp -r /path/source root@192.168.10.251:/data/models

copy到远程服务器

## Move a file/folder

mv /path/source /path/to

## Check size

`du -sh /path/to/folder`

-s: summarizes the total size of the folder instead of listing the size of each individual file within the folder.
-h: human-readable format, showing the size in KB, MB, GB, etc., for easier understanding.
