import random
import sys
prodict={}
with open(sys.argv[1]) as f:
	for line_id, line in enumerate(f):
		if line[0]=='>':
			if line_id==0:
				pid=line
				pro=''
			else:
				prodict[pid]=pro
				pid=line
				pro=''
		else:
			pro+=line.strip()
prodict[pid]=pro

with open(sys.argv[2],'w') as f:
	for line in prodict:
		f.write(line)
		f.write(prodict[line]+'\n')
		f.write('>shuffle_'+line[1:])
		f.write(''.join(random.sample(prodict[line],len(prodict[line])))+'\n')