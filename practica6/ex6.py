from practica6 import TopK

if __name__=='__main__':
    a=TopK('ex6data.txt',20)
    solucion=['set([])']
    solucion.append('set([\'Female\'])')
    solucion.append('set([\'DoNotOwnHome\'])')
    solucion.append('set([\'Homeowner\'])')
    solucion.append('set([\'Male\'])')
    solucion.append('set([\'cannedveg\'])')
    solucion.append('set([\'frozenmeal\'])')
    solucion.append('set([\'fruitveg\'])')
    solucion.append('set([\'beer\'])')
    solucion.append('set([\'fish\'])')
    solucion.append('set([\'wine\'])')
    solucion.append('set([\'confectionery\'])')
    solucion.append('set([\'Female\', \'Homeowner\'])')
    solucion.append('set([\'Female\', \'DoNotOwnHome\'])')
    solucion.append('set([\'Male\', \'DoNotOwnHome\'])')
    solucion.append('set([\'Male\', \'Homeowner\'])')
    solucion.append('set([\'Male\', \'frozenmeal\'])')
    solucion.append('set([\'cannedmeat\'])')
    solucion.append('set([\'cannedveg\', \'Male\'])')
    solucion.append('set([\'fish\', \'DoNotOwnHome\'])')
    solucion.append('set([\'beer\', \'Male\'])')
    for num,e in enumerate(a):
        print "Solucion:",solucion[num]," Valor del programa:",e
