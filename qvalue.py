def qvalue(data):
	fp=0
	tp=0
	flagtp=[]
	for x in data: 
		if x[4]=='True': 
			tp+=1 
		else: 
			fp+=1 
		flagtp.append([fp,tp]) 
	for line_id in range(len(flagtp)):
		data[line_id].append(str(flagtp[line_id][0]/flagtp[line_id][1]))

	return data
	

data=[]
with open('fixed_test/benchmark/test.tsv') as f: 
	for line in f: 
		s=line.strip().split('\t') 
		data.append([s[0],s[1],s[2],s[3],s[4],s[5]])

sdata=sorted(data,key=lambda x:x[0],reverse=True)
dfq=qvalue(sdata)
scomet=sorted(dfq,key=lambda x:x[1],reverse=True)
cometq=qvalue(scomet)
smsgf=sorted(cometq,key=lambda x:x[2],reverse=True)
msgfq=qvalue(smsgf)
smyrimatch=sorted(msgfq,key=lambda x:x[3],reverse=True)
myrimatchq=qvalue(smyrimatch)
with open('fixed_test/benchmark/test_qvalue.tsv','w') as f:
	for line in myrimatchq:
		f.write('\t'.join(line)+'\n')