
fvar = open ("animate.py","r")
fout = open("strip_comments_"+"animate.py","w")

for line in fvar:

	if line.find('#') != -1:
		newline = line[:line.find('#')]
	else:
		newline = line[:-1]

	fout.write(newline+'\n')

fvar.close()
fout.close()