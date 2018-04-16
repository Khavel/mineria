#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Teneis que subir un script en python que, leyendo este fichero como texto
y utilizando expresiones regulares encuentre:

- El nombre del matemático
- El nombre de su director de tesis (advisor)
- El nombre de todos los alumnos que ha dirigido
"""

import sys
import urllib.request
url = 'http://genealogy.math.ndsu.nodak.edu/id.php?id=36415'
response = urllib.request.urlopen(url)
data = response.read()      # a `bytes` object
text = data.decode('utf-8') # text!

import re


#Matematico

m = re.search(r'',text)
print( "\nNombre del matematico : "+m)


#Advisor '?<=Advisor: \w+'
a = re.search(r'',text)
print ("Nombre del director de tesis (advisor) : "+a)


#Alumnos
alus = re.search(r'',text)
print("Nombre los alumnos : " + str(alus))
