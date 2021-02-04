
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Enzo
"""
import numpy as np 
import math 
from Rbm import Rbm
from tqdm import tqdm


class Dnn:
    """
    Implémentation d'un DNN avec des sigmoides comme fonctions d'activation sur les couches cachées et softmax 
    en couche de sortie 
    
    Methods
    ----------------
    __init__:
        Object constructor
    pretrain:
        Pré entrainement du réseau
    genere_image:
        Genere des images reconstituée par le réseau vu comme un Dbn
    entree_sortie_reseau:
        Calcule l'état des unités du réseau
    retropropagation:
        Entraine le réseau par descente de gradient
    test:
        Teste les performances du réseau appris
    
    Attributes
    ---------------
    size:
        une liste ou l'élément i représente le nombre d'unité sur la couche i
    rbm_list:
        une liste de rbm représentant les poids du réseau       
    
    """
    
    def __init__(self, size):
        """
        Object constructor

        Parameters
        ----------
        size : list
            une liste ou l'élément i représente le nombre d'unité sur la couche i

        Returns
        -------
        None.

        """
        self.size = size
        self.rbm_list = []
        for i in range(len(size)-1):
            self.rbm_list += [Rbm(size[i],size[i+1])]
    
    def pretrain(self,data,epoch,learning_rate,mini_batch_size):
        """
        Entraine le réseau à la manière d'un Dbn sur les n-1 premières couches

        Parameters
        ----------
        data : np.ndarray
            les données
        epoch : int
            nombre d'itération
        learning_rate : float
        mini_batch_size : int

        Returns
        -------
        None.

        """
        #on copie les données par sécurité
        data_train = data.copy()
        #entrainement non supervisé des n-1 rbm
        _ = 1
        for rbm in self.rbm_list[:len(self.rbm_list)-1]:
            print('train rbm ',_)
            _+=1
            rbm.train(data_train,epoch,learning_rate,mini_batch_size)
            data_train = (np.random.uniform(size=(len(data_train),rbm.nh)) < rbm.entree_sortie(data_train)).astype(int)
            
    def genere_images(self,n,iter_gibbs):
        """
        genere des images reconstituée par le réseau vu comme un Dbn

        Parameters
        ----------
        n : int
            nombre d'images que l'on souhaite générer
        iter_gibbs : int
            nombre d'itération du gibbs sampler

        Returns
        -------
        V : list
            liste d'images générées

        """
        V = []
        for i in range(n):
            #donnée aléatoire
            v = np.random.binomial(1,0.5,size=self.size[0])
            #gibbs sampling sur les n-1 couches
            for j in range(iter_gibbs):
                #aller
                for rbm in self.rbm_list[:len(self.rbm_list)-1]:
                    v = (np.random.uniform(size=rbm.nh) < rbm.entree_sortie(v)).astype(int)
                #retour
                for rbm in reversed(self.rbm_list[:len(self.rbm_list)-1]):
                    v = (np.random.uniform(size=rbm.nv) < rbm.sortie_entree(v)).astype(int)
            V+=[v]
        return V
    
    def entree_sortie_reseau(self,V):
        """
        Calcule l'état des unités du réseau

        Parameters
        ----------
        V : np.ndarray
            liste de données 

        Returns
        -------
        X : TYPE
            liste des états des unités des différentes couches.

        """
        
        X = [V]
    
        for i in range(len(self.rbm_list)-1):
            rbm = self.rbm_list[i]
            V = rbm.entree_sortie(V)
            X+=[V]
        rbm = self.rbm_list[-1]
        X += [rbm.calcul_softmax(V)]
        return X
    
    def retropropagation(self,data,labels,epoch,learning_rate,mini_batch_size):
        """
        Entraine le réseau par descente de gradient

        Parameters
        ----------
        data : np.ndarray
            Données
        labels : np.ndarray
            labels one hot encodé
        epoch : int 
        learning_rate : float
        mini_batch_size : int

        Returns
        -------
        None.

        """
        data_train = data.copy()
        labels_train = labels.copy()
        
        for e in tqdm(range(epoch)):
            #shuffle des données
            indices = np.arange(data_train.shape[0])
            np.random.shuffle(indices)
            
            data_train = data_train[indices]
            labels_train = labels_train[indices]

            #creation des batchs
            data_batchs = [data_train[i:i+mini_batch_size] for i in range(0,len(data),mini_batch_size)]
            labels_batchs = [labels_train[i:i+mini_batch_size] for i in range(0,len(data),mini_batch_size)]
            
            cross_entropy = 0 
            
            
            for batch in zip(data_batchs,labels_batchs): 
                
                X = self.entree_sortie_reseau(batch[0])
                Y = batch[1]
                
                
                cross_entropy -= (np.log(X[-1])*Y).sum()

                delta = X[-1]-Y
                               
                for i in range(len(self.rbm_list)-1,-1,-1):
                    delta_W =  X[i].T @ delta
                    delta_b = delta.sum(axis=0)
                    
                    if i > 0:
                        #dérivée de la fonction d'activation
                        sigma_deriv = X[i]*(1-X[i])
                        #update des delta
                        delta = sigma_deriv* ( delta @ self.rbm_list[i].W.T)
                        
                    self.rbm_list[i].W -= learning_rate*delta_W/len(batch)
                    self.rbm_list[i].b -= learning_rate*delta_b/len(batch)
            print("cross entropie : ")
            print(cross_entropy/len(data))
          
    def test(self,data,labels):
        """
        teste les performances du réseau

        Parameters
        ----------
        data : np.ndarray
            données de test
        labels : np.ndarray
            labels de test one hot encodés

        Returns
        -------
        error_rate : float
            le taux d'érreur sur les données de test

        """

        batch_size = math.ceil(len(data)/10)
        data_batchs = [data[i:i+batch_size] for i in range(0,len(data),batch_size)]
        labels_batchs = [labels[i:i+batch_size] for i in range(0,len(data),batch_size)]
        
        error = 0
        for batch in zip(data_batchs,labels_batchs):
            X = self.entree_sortie_reseau(batch[0])[-1]
            Y = batch[1]
            result = np.zeros_like(X)
            #le résultat de notre classification est le maximum de notre dernière couche d'ou la ligne suivante
            result[np.arange(len(X)),X.argmax(axis = 1)] = 1
            
            error += np.abs(result - Y).sum()/2
                      
        print("error rate: ", error/len(data) )
        return error/len(data) 
        
                    
                
                
            
            
        
            
                
            
            
    