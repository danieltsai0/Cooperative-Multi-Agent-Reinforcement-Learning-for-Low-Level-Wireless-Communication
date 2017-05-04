# deeprl-project

## Structure

* /shanes_work - shanes stuff (private work-folder)
* /colins_work - colins stuff (private work-folder)
* /base_implementation - fixed receiver, transmitter
* /decentralized_implementation - decentralized approach
* /plots - archive for cool gifs and plots to be used in the future

***
## Structure Final Report
1.  Introduction  
    Applying learning algorithms to classic radio tasks
1.  Related Work
    * Foerster - multi-agent but centralized learning
    * OpenAI - multi-agents learning comms but differentiable
    * Google Brain - Learning coding, mutual information, but shared gradients, adverserial setting
    * some papers on network parameter tuning with RL
    * ...?
2.  Background Information
    1.  Low Level Wireless Communication  
        Complex signals, modulation, classic modulation schemes, Gray coding, perfomance plots
    2.  Reinforcement Learning  
        Markov Decision Process, optimization problem, policy gradients, vanilla score function gradient estimator
3.  Preliminary Analysis (Data driven approaches to classic radio tasks)
    1.  Modulation Recognition  
        Eb/N0 Plots over #training samples 
        -> Result: receiving is easy, so keep it easy
    2.  Symbol Timing Recovery (How to evaluate?) 
    3.  Equalization
    4.  Training a single agent  
        Only transmitter, since receiver is easy
4.  Problem formulation  
    Decentralized, multi-agent learning of modulation
5.  Problem setup  
    Two agents, shared preamble, transmitter and receiver architecture, echoing, no other shared information
6.  Results  
    **Two evaluation methods:**
    *   loss plot: L1 difference between b^^ and b over training
    *   performance plot: BER over EbN0 after training  
    
 
    1.  Unrestricted power during learning
        * preamble BER during training (loss-plot)
        * BER over Eb/N0 after learning (performance plot)   
        * development of Eb
    2.  Restricted power during learning
        * preamble BER during testin (loss-plot)
        * BER of Eb_N0 after learning (performance plot)

***
## Deadlines

4/11: fixed Tx/Rx
4/17: one page description due at beginning of class
5/8: 5-8 page report


## 4/4/17:

look at apsk, bpsk, qpsk, 16-quam do we want to whether we want to paramaterize output of transmitter as cartesian or polar?  
fixed Tx, learn Rx:
⋅⋅⋅input x,y of complex, softmax output + eps greedy / boltzman exploration


## 4/18/17:
tasks:
1. reward shaping for transmitter (need to restrict power and maximize distance between points. former must be stronger than latter to prevent outer points from flying away)

2. Tx -> Rx and Rx gives reward back to Tx. Rx provides k-nn guess for each datasample back to Tx
3. Tx on both sides, Rx on both sides. 
4. OpenAI style

![Alt text](https://c1.staticflickr.com/3/2929/33290540844_2afbbcd75d_b.jpg )



## Reports
### Final Report
[https://www.sharelatex.com/project/58fe82f296da09b1289caec3](https://www.sharelatex.com/project/58fe82f296da09b1289caec3)
### CS294-121: 
[https://www.sharelatex.com/project/58eedf885eecccdc7a8817a8](https://www.sharelatex.com/project/58eedf885eecccdc7a8817a8)
### CS294-133:
[https://www.sharelatex.com/project/58bf4907d9f1e6e906db8ad5](https://www.sharelatex.com/project/58bf4907d9f1e6e906db8ad5)
