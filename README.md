# PendulumOpenAi
Criação de um algoritmo de redes neurais usando Tensor Flow em Python.
## Equipe 
Gustavo José Salvalaggio
## Problema
O problema do pendulo é clássico de literatura. O pendulo começa em uma posição aleatória, e o objetivo é balança-lo até que ele fique em posição vertical apontado para cima. 
é necessário utilizar vetores de força em eixo y para fazer o pendulo balançar, treinando um modelo para aplicação de força necessária para que o pendulo fique apontado para cima.
## Dataset
Dataset pronto da https://gym.openai.com/envs/Pendulum-v0/.
O gym é um kit de ferramentas para desenvolver e comparar algoritmos de aprendizado por reforço.
## Técnica
Foi utilizado um treinamento através de redes neurais utilizando a técnica de aprendizagem por reforço.

Ao trabalhar com o gym, exis  tem 4 valores que precisamos enviar e trabalhar para fazer o algoritmo funcionar. eles são:

observation: um objeto específico do ambiente que representa a observação do ambiente. No nosso caso, são valores de intervalo de força para as direçoes.
reward: recompensa alcançada pela ação anterior. O nosso objetivo foi aumentar sua recompensa total.
done: se é hora reset do ambiente novamente. Neste desafio não possui um tarefa com episódio bem definifo, pois é usado uma iteração para encerrar o episodio, já que não existe condição de vitória, apenas o estado do pendulo o mais vertical possível.
info: informações de diagnóstico úteis para depuração. Apesar de trazer informações de diagnóstico do que foi realizado, não utilizei para o algoritmo de treinamento feito.

![image](https://user-images.githubusercontent.com/45314777/154880425-023643b8-80b4-42a2-b2a9-436565918544.png)

## Resultados Obtidos
Foi criado um script para treinamento do modelo, utilizando 1000 iterações com 25 intervalo de testes. Levaram alguns dias até conseguir um resultado em que o pendulo se mantesse de pé de um jeito aceitável, já que fui anotando os valores de teste e testando de acordo com a performance ao finalizar o treinamento. Para treinar um modelo com os parâmetros atuais, leva em torno de umas 4 ~ 6 horas por vez.
O modelo no projeto é o modelo com a melhor eficiencia nos testes.
## Instruções de uso do software
com o python instalado, rodar o comando 
```
pip install tensorflow
```
e em seguida, o comando 
```
pip install gym
```
com isso, todas as bibliotecas e ferramentas necessárias para o sistema estará funcionando.
a partir disso, abrir um prompt no local dos arquivos e rodar o comando
```
python main.py
```
OBS:
foi criado um arquivo executável, mas por algum motivo o tensorflow não funcionava direito, fazendo o tamanhano ficou maior que 100mb, fazendo com que eu não conseguisse dar um push para o git.
