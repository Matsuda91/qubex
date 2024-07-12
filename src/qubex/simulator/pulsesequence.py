import numpy as np
import itertools
import matplotlib.pyplot as plt
from tabulate import tabulate
from .system import System


SAMPLING_PERIOD = 0.1 # ns

class Pulse:
    
	def __init__(self, amplitude,  phase, shape):
		self.amplitude = amplitude
		self.phase = phase
		self.shape = shape
		self._validate_shape()
	
	def _validate_shape(self):
		intensity = np.abs(self.shape/self.amplitude)
		if np.max(intensity) > 1:
			raise ValueError("Pulse shape intensity has values exceeding 1. The range is [-1,1]")
		else:
			pass

class RaisedCosFlatTop(Pulse):

	def __init__(
			self,
			label,
			amplitude,
			phase,
			duration,
			risetime,
			):
		self.amplitude: float = amplitude
		self.phase: float = phase
		self.label:str =label
		self.duration: float = duration
		self.risetime: float = risetime
		self.shape:np.ndarray = self._raised_cos()
		super().__init__(self.amplitude, self.phase, self.shape)

	def _raised_cos(self):
		t = np.linspace(0, self.duration-SAMPLING_PERIOD, int(self.duration/SAMPLING_PERIOD))
		shape = np.zeros(len(t))
		
		shape[t <= self.risetime] = 0.5*(1-np.cos(np.pi*t[t <= self.risetime]/self.risetime))
		shape[t > self.duration-self.risetime] = 0.5*(1-np.cos(np.pi*(t[(t > self.risetime) & (2*self.risetime > t)]/self.risetime)))
		shape[(t > self.risetime) & (t <= self.duration-self.risetime)] = 1
		return self.amplitude*shape*np.exp(1j*self.phase)


class Gaussian(Pulse):

	def __init__(
			self,
			label,
			amplitude, 
			phase,  
			duration):
		self.amplitude: float = amplitude
		self.phase: float = phase
		self.label:str =label
		self.duration: float = duration
		self.n_sigma:int = 3
		self.shape:np.ndarray = self._gaussian()
		super().__init__(self.amplitude, self.phase, self.shape)
		
	def _gaussian(self):
		t = np.linspace(0, self.duration-SAMPLING_PERIOD, int(self.duration/SAMPLING_PERIOD))
		sigma = self.duration/2
		shape = np.exp(-0.5*((t-self.duration/2)**2)/((sigma/self.n_sigma)**2)) 
		return self.amplitude*shape*np.exp(1j*self.phase)

class Half_pi(Pulse):

	def __init__(
			self,
			label,
			amplitude,
			phase,
			duration,
	):
		self.amplitude: float = amplitude
		self.phase: float = phase
		self.label:str =label
		self.duration: float = duration
		self.shape:np.ndarray = self._raised_cos()
		super().__init__(self.amplitude, self.phase, self.shape)

	def _raised_cos(self):
		t = np.linspace(0, self.duration-SAMPLING_PERIOD, int(self.duration/SAMPLING_PERIOD))


class Sequence:
	def __init__(self,system: System):
		self.system = system
		self.channels = self._channels()
		self.sequences = self._init_sequences()
		self.padding_timestamp = []
		
		self._check_sampling_period()

	def _check_sampling_period(self):
		frequencies = []
		for _, transmon in enumerate(self.system.transmons):
			frequencies.append(transmon.frequency)
		diff = []
		for _, combination in enumerate(itertools.combinations_with_replacement(frequencies, 2)):
			diff.append(np.abs(combination[0]-combination[1]))
		if 1/np.max(diff)<SAMPLING_PERIOD:
			raise ValueError(f"Sampling period is too large. The maximum frequency difference is {np.max(diff)}")
		else:
			pass
	
	def add_pulse(self, pulses: list[Pulse], flush:str):
		for _, pulse in enumerate(pulses):
			self.sequences[pulse.label] = np.append(self.sequences[pulse.label], pulse.shape)
		self._padding(flush)

	def _padding(self,flush):
		labels = self.channels.keys()
		lengthes = [len(self.sequences[label]) for _, label in enumerate(labels)]
		max_length = np.max(lengthes)
		if flush == 'left':
			for kdx,label in enumerate(labels):
				self.sequences[label] = np.pad(self.sequences[label], (0,int(max_length - lengthes[kdx])))
		else:
			if flush != 'right':
				raise ValueError("flush must be either 'left' or 'right'")
			for kdx,label in enumerate(labels):
				self.sequences[label] = np.pad(self.sequences[label], (int(max_length - lengthes[kdx]),0))
		self.padding_timestamp.append(max_length*SAMPLING_PERIOD)

	def _init_sequences(self):
		sequences = {}
		for _, channel in enumerate(self.channels.keys()):
			sequences[channel] = np.array([])
		return sequences
	
	def _channels(self) -> dict[str, float]:
		def switch(label):
			index = label.index('-')
			return label[index+1:]+'-'+label[:index]
		channels = {}
		for _, transmon in enumerate(self.system.transmons):
			channels[transmon.label] = transmon.frequency
			for _, coupling in enumerate(self.system.couplings):
				if  "-" in coupling.label:
					
					index1 = list(self.system.graph.nodes).index(coupling.pair[1])
					channels[coupling.label] = self.system.transmons[index1].frequency
					
					index0 = list(self.system.graph.nodes).index(coupling.pair[0])
					channels[switch(coupling.label)] = self.system.transmons[index0].frequency
				else:
					pass
		return channels
	
	@property
	def channels_information(self):
		info = {'labels': self.channels.keys(), 'frequencies (GHz)': self.channels.values()}
		print(tabulate(info, headers='keys', tablefmt='simple'))
	
	@property
	def times(self) -> np.ndarray:
		sequence_length = self.padding_timestamp[-1]
		return np.linspace(0, sequence_length-SAMPLING_PERIOD, int(sequence_length/SAMPLING_PERIOD))
	
	def control_qubit_2Qgate(self,label) -> str:
		index = label.index('-')
		return label[:index]
	
	def target_qubit_2Qgate(self,label) -> str:
		index = label.index('-')
		return label[index+1:]
	
	def plot(self,yauto=True):
		fig = plt.figure(figsize=(float(self.times[-1]*0.1),0.8*len(self.sequences.keys())))
		for ldx, label in enumerate(self.sequences.keys()):
			plt.subplot(len(self.sequences.keys()),1,ldx+1)
			plt.tick_params(axis='x', labelbottom=False)
			plt.step(self.times,np.real(self.sequences[label])/(2*np.pi)*1e3,zorder=1)
			plt.step(self.times,np.imag(self.sequences[label])/(2*np.pi)*1e3,zorder=0)
			for timestamp in self.padding_timestamp:
				plt.axvline(x=timestamp, color='r', linestyle='--')
			plt.text(0.05, 0.75, label, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
			plt.ylabel('(MHz)')
			if yauto:
				pass
			else:
				plt.ylim(-1.1,1.1)
		plt.tick_params(axis='x', labelbottom=True)
		plt.xlabel('times (ns)')

		plt.show()