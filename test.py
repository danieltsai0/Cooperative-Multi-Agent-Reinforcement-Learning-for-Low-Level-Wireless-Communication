class Test:
	def __init__(self, state):
		self.one = "rh3"
		self.two = "2444"
		self.func = None

		if state:
			self.func = self.func1
		else:
			self.func = self.func2

	def do_stuff(self, arg):
		print(self.func(arg))

	def func1(self, phrase):
		return self.one+phrase

	def func2(self, phrase):
		return self.two+phrase


t1 = Test(0)
t2 = Test(1)

t1.do_stuff("hello")
t2.do_stuff("hello")