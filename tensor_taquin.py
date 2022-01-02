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


def move_possible(tab, precedent=-1):
    n  = int(sqrt(len(tab)))
    pos = pos_vide(tab)
    choixPossibles = []
    if pos >= n and precedent != 1 :            choixPossibles.append(0)
    if pos < len(tab) - n and precedent != 0 :  choixPossibles.append(1)
    if pos%n < n-1 and precedent != 3:          choixPossibles.append(2)
    if pos%n > 0 and precedent != 2 :           choixPossibles.append(3)
    return choixPossibles


def creation_donnees(repetition, profondeurPartie, n):
    train_parties, train_move = [], []
    move = -1
    for i in range(repetition):
        partie = init(n)
        pos = 15
        for j in range(profondeurPartie):
            move = choice(move_possible(partie, move))
            partie, pos = jeu(partie, move, pos)
            goal = [0, 0, 0, 0]
            goal[[1, 0, 3, 2][move]] = 1
            train_parties.append(partie.copy())
            train_move.append(goal)
    return np.array(train_parties), np.array(train_move)


def melange_jeu(tab, profondeur, pos):
    for i in range(profondeur):
        tab, pos = jeu(tab, choice(move_possible(tab)), pos)
    return tab, pos


def jeu(tab, choix, pos):
    if choix in move_possible(tab):
        if choix == 0:
            n  = int(sqrt(len(tab)))
            tab[pos], tab[pos-n] = tab[pos-n], tab[pos]
            pos -= n
        elif choix == 1:
            n  = int(sqrt(len(tab)))
            tab[pos], tab[pos+n] = tab[pos+n], tab[pos]
            pos += n
        elif choix == 2:
            tab[pos], tab[pos+1] = tab[pos+1], tab[pos]
            pos += 1
        elif choix == 3:
            tab[pos], tab[pos-1] = tab[pos-1], tab[pos]
            pos -= 1
    return tab, pos

def train():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(16,)))
    model.add(tf.keras.layers.Dense(256, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(128, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(128, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(64, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(64, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(64, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(16, activation=tf.math.sigmoid))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[])


    for i in range(20):
        print("n°",i)
        inData, outData = creation_donnees(10000, 20, 4)
        model.fit(inData, outData, epochs=2)

    return model

def simulation(model, melange):
    tableau, fin = init(4), init(4)
    tableau, pos = melange_jeu(tableau, melange, 15)
    print(tableau, "\n")
    while not np.array_equal(tableau, fin):
        prediction = model.predict(tableau.reshape(-1,16))
        move = prediction.argmax()
        tableau, pos = jeu(tableau, move, pos)
        affichage_tableau(tableau)
        print()


model = tf.keras.models.load_model("modelTaquin")