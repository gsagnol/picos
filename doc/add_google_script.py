import os
files = os.popen('ls full_html/last/*.html').readlines()
files = [f[:-1] for f in files if f[:6] not in ('search','py-mod','genind')]

for f in files:
        fi=open(f,'r')
        fitmp=open(f+'tmp','w')
        line = fi.readline()
        while '</head>' not in line:
                fitmp.write(line)
                line = fi.readline()

        fitmp.write('\n')
        fitmp.write('\n')
        fitmp.write('  <script type="text/javascript">\n')
        fitmp.write('\n')
        fitmp.write('  var _gaq = _gaq || [];\n')
        fitmp.write("  _gaq.push(['_setAccount', 'UA-33037163-1']);\n")
        fitmp.write("  _gaq.push(['_trackPageview']);\n")
        fitmp.write('\n')
        fitmp.write("  (function() {\n")
        fitmp.write("    var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;\n")
        fitmp.write("    ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';\n")
        fitmp.write("    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);\n")
        fitmp.write("  })();\n")
        fitmp.write('\n')
        fitmp.write("  </script>\n")
        fitmp.write('\n')
        while line:
                if 'tar.gz' in line:
                        vsplit=line.split('tar.gz')
                        version=vsplit[0].split('PICOS')[1][1:-1]
                        vsplit=version.split('.')
                        ocstring=('''onClick="javascript: _gaq.push(['_trackPageview', '/downloads/version'''
                                 +str(vsplit[0]) +str(vsplit[1]) +str(vsplit[2])
                                 + '''']);"''')
                        indtar = line.index('tar.gz') + 7
                        line = line[:indtar]+' '+ocstring+line[indtar:]
                fitmp.write(line)
                line = fi.readline()
                
        fi.close()
        fitmp.close()
        os.system('mv '+f+'tmp '+f)
        
