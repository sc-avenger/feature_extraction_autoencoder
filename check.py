f1="HelloWorld;"
f=open("check.txt","w+")
class Hello:
	def printg(self):
		f.write(str(f1))
		
h = Hello()
h.printg()
for i  in range(100000):
		print(i)


f.close()