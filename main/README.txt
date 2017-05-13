system.py: 
	contains: System class
	purpose: to define an environment in which two Actor instances can
			 simulate interactions and learning

actor.py: 
	contains: Actor class
	purpose: a wrapper for a transmitter and receiver unit, represents
			 a single radio actor in the System

transmitter.py:
	contains: NeuralTransmitter class
	purpose: class for transmitter unit

receiver.py:
	contains: KnnReceiver class
	purpose: class for receiver unit

channel.py:
	contains: Channel class
	purpose: class for simulating channel noise

evaluate.py:
	contains: script to evaluate the performance of a transmitter

util.py:
	contains: utility methods