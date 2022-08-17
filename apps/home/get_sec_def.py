def getDef(secNo):

    with open('apps/home/indian-penal-code_final.csv') as f:
        mylist = list(f)
    mylist=" ".join(mylist)

    d=secNo
    j="Section "+str(d)
    k="Section "+str(d+1)

    a, b = mylist.find(j), mylist.find(k)
    output=mylist[a+len(j):b]

    return output