import tensorflow as tf
import numpy as np
from random import choice
from math import sqrt


def init(n):
    return np.array([(i+1)%n**2 for i in range(n**2)])


def affichage_tableau(tab):
    n = int(sqrt(len(tab)))
    for i in range(n):
        for j in range(n):
            print(tab[i*n+j], end="\t")
        print()


def pos_vide(tab):
    return np.where(tab == 0)[0][0]


def move_possible(tab):
    n  = int(sqrt(len(tab)))
    pos = pos_vide(tab)
    choixPossibles = []
    if pos >= n : choixPossibles.append(0)
    if pos < len(tab) - n : choixPossibles.append(1)
    if pos%n < n-1 : choixPossibles.append(2)
    if pos%n > 0 : choixPossibles.append(3)
    return choixPossibles


def creation_donnees(repetition, profondeurPartie, n):
    train_parties, train_move = [], []
    for i in range(repetition):
        partie = init(n)
        for j in range(profondeurPartie):
            move = choice(move_possible(partie))
            partie = jeu(partie, move)
            train_parties.append(partie)
            goal = [0, 0, 0, 0]
            goal[[1, 0, 3, 2][move]] = 1
            train_move.append(goal)
    return np.array(train_parties), np.array(train_move)


"""
init -> la grille de base

creation données samples_num partie_depth
    train_x = []
    train_y = []
    for i in range samples_num//partie_depth
        partie = init()
        for p in range partie_depth
            move = random 0-3
            
            train_x += [partie]
            goal = [0, 0, 0, 0]
            goal[[1, 0, 3, 2][move]] = 1
            train_y += [goal]

            modifie partie move
"""

def melange_jeu(tab, profondeur):
    for i in range(profondeur):
        tab = jeu(tab, choice(move_possible(tab)))
    return tab


def jeu(tab, choix):
    pos = pos_vide(tab)
    if choix in move_possible(tab):
        if choix == 0:
            n  = int(sqrt(len(tab)))
            tab[pos], tab[pos-n] = tab[pos-n], tab[pos]
        elif choix == 1:
            n  = int(sqrt(len(tab)))
            tab[pos], tab[pos+n] = tab[pos+n], tab[pos]
        elif choix == 2:
            tab[pos], tab[pos+1] = tab[pos+1], tab[pos]
        elif choix == 3:
            tab[pos], tab[pos-1] = tab[pos-1], tab[pos]
    return tab


model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(16,)))
model.add(tf.keras.layers.Dense(128, activation=tf.math.sigmoid))
model.add(tf.keras.layers.Dense(64, activation=tf.math.sigmoid))
model.add(tf.keras.layers.Dense(32, activation=tf.math.sigmoid))
model.add(tf.keras.layers.Dense(16, activation=tf.math.sigmoid))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[])

inData, outData = creation_donnees(5000, 20, 4)

model.fit(inData, outData, epochs=10)

for a,b in zip(inData, outData):
    affichage_tableau(a)
    print(b)
    print()


def simulation():
    tableau, fin = init(4), init(4)
    tableau = melange_jeu(tableau, 40)
    print(tableau)
    while not np.array_equal(tableau, fin):
        prediction = model.predict(tableau.reshape(-1,16))
        move = prediction.argmax()
    
        if move in move_possible(tableau):
            tableau = jeu(tableau, move)
            print(tableau)
        else:
            print()
            affichage_tableau(tableau)
            print("ERREUR !")
            print(prediction)
            print()
            break


simulation()