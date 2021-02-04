#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Enzo
"""
import numpy as np
import scipy.special
from tqdm import tqdm

class Rbm:

    """
    
    Implémentation d'une machine de Boltzmann restreinte 
    
    Methods
    -------------
    __init__:
        Object constructor 
    entree_sortie:
        donne la probabilité que les unités cachées valent 1 sachant l'état des unités visibles
    sortie_entree:
        donne la probabilité que les unités visibles valent 1 sachant l'etat des unités cachées
    train:
        entraine le rbm suivant la méthode CD1
    genere_images :
        génère une liste d'images reconstruites par le rbm
        
    Attributes
    ------------
    W : numpy array, size = (nv,nh)
        weight of the rbm
    a : numpy array, size = (nv)
        bias 
    b: numpy array, size = (nh)
    
    """
    
    def __init__(self,nv,nh):
        """
        Initialise le rbm

        Parameters
        ----------
        nv : int
            nombre d'unités sur la couche visible.
        nh : int
            nombre d'unités sur la couche cachée.

        Returns
        -------
        None.

        """
        
        self.nh = nh
        self.nv = nv
        self.W = np.random.normal(scale = 0.1,size=(nv,nh))
        self.b = np.zeros(nh)
        self.a = np.zeros(nv)


    def entree_sortie(self, V):
        """
        Donne la probabilité que les unités cachées valent 1 sachant l'état des unités visibles

        Parameters
        ----------
        V : np.ndarray 
            tableau numpy qui contient une liste d'état de la couche visible

        Returns
        -------
        PHV : np.ndarray
            tableau qui contient pour chaque état de la couche visible donné en paramètre les probabilités
            que les unités de la couche cachée valent 1

        """
        PHV = scipy.special.expit( self.b + V@self.W)
        return PHV

    def sortie_entree(self, H):
        """
        Donne la probabilité que les unités cachées valent 1 sachant l'état des unités visibles

        Parameters
        ----------
        V : np.ndarray 
            tableau numpy qui contient une liste d'état de la couche visible

        Returns
        -------
        PHV : np.ndarray
            tableau qui contient pour chaque état de la couche visible donné en paramètre les probabilités
            que les unités de la couche cachée valent 1

        """
        
        PVH = scipy.special.expit(self.a+ H@self.W.T)
        return PVH
    
    def train(self,data,epoch,learning_rate,mini_batch_size):
        """
        entraine le rbm selon CD 1

        Parameters
        ----------
        data : np.ndarray
            données d'entrées.
        epoch : int
            nombre d'iterations.
        learning_rate : float
        mini_batch_size : int

        Returns
        -------
        None.

        """
        data_train = data.copy()
        
        for e in tqdm(range(epoch)):
            #shuffling the data
            np.random.shuffle(data_train)
            #creating batches
            batchs = [data_train[i:i+mini_batch_size] for i in range(0,len(data_train),mini_batch_size)]
            for batch in batchs:
                
                PHV  = self.entree_sortie(batch)
                #CD1
                U = np.random.uniform(size=(len(batch),self.nh))     
                H_gibbs = (U < PHV).astype(int)
                
                U = np.random.uniform(size=(len(batch),self.nv))
                V_gibbs = (U < self.sortie_entree(H_gibbs)).astype(int)
                PHV_gibbs = self.entree_sortie(V_gibbs)
                
                #gradient computation
                delta_W = batch.T @ PHV - V_gibbs.T @ PHV_gibbs
                delta_a = (batch - V_gibbs).sum(axis = 0)
                delta_b = (PHV - PHV_gibbs).sum(axis=0)
                
                #weights update 
                
                self.W += learning_rate*delta_W/len(batch)
                self.a += learning_rate*delta_a/len(batch)
                self.b += learning_rate*delta_b/len(batch)
            
            #calcul de l'erreur quadratique
            #reconstruction
            PHV  = self.entree_sortie(data_train)
            U = np.random.uniform(size=(len(data_train),self.nh)) 
            Hidden = (U < PHV).astype(int)
            U = np.random.uniform(size=(len(data_train),self.nv))
            data_rec = (U < self.sortie_entree(Hidden)).astype(int)
            #resultat
            r2  = ((data_train-data_rec)**2).sum()
            print('erreur quadratique : ',r2)
            
            
                
                
                
    def generer_image(self,n,iter_gibbs):
        """
        génère une liste d'images reconstruites par le rbm

        Parameters
        ----------
        n : int
            Nombre d'images souhaitées.
        iter_gibbs : int
            Nombres d'itération du gibbs sampler
        Returns
        -------
        V : list 
            Liste des images régénérées

        """
        V = []
        for i in range(n):
            v = np.random.binomial(1,0.5,size=self.nv)
            #Gibbs sampling
            for j in range(iter_gibbs):
                u = np.random.uniform(size=self.nh)
                h = u < self.entree_sortie(v)
                
                u = np.random.uniform(size=self.nv)
                v = u < self.sortie_entree(h)
            V+=[v]
        return V
    
    def calcul_softmax(self,V):
        return scipy.special.softmax(V@self.W+self.b,axis = 1)
    
                
                    
                    
                    
                    
            
        
    