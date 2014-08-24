newlines =['<body>',
           '<div class="admonition warning">',
           '<p class="first admonition-title">Warning</p>',
           '<p class="last">You are consulting the doc of a former version of PICOS.',
           'The latest version is <a href="../index.html">HERE</a>. </p>',
           '</div>'
           ]

import os
files = os.popen('ls *.html').readlines()
files = [f[:-1] for f in files if f[:6] not in ('search','py-mod','genind')]

for f in files:
        fi=open(f,'r')
        fitmp=open(f+'tmp','w')
        line = fi.readline()
        while '<body>' not in line:
                fitmp.write(line)
                line = fi.readline()

        for ln in newlines:
                fitmp.write(ln)
                
        line = fi.readline()
        while line:
                fitmp.write(line)
                line = fi.readline()
                
        fi.close()
        fitmp.close()
        os.system('mv '+f+'tmp '+f)
        
