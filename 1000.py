import codecs 
with codecs.open("wiki.en.text",encoding="utf-8") as f, codecs.open("demo.txt","w",encoding="utf-8") as out:
	for index,line in enumerate(f):
		out.write(line)
		if index>1000:
			break
