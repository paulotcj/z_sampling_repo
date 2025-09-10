import numpy as np
import matplotlib.pyplot as plt
import math

# input is an array of complex numbers. We take the real and imaginary part and apply the absolute value of the complex number to
#   each of the elements of the array
#   -> SQRT(A^2 + B^2)
def AbsoluteValueForComplexNumbers(input):
    print("AbsoluteValueForComplexNumbers")
    returnObject = np.array([0.0] * input.size)
     
    for x in range(0,input.size):
        returnObject[x] = math.sqrt( input[x].real**2 + input[x].imag**2 )

    return returnObject
    

# This is the discrete Fourier transform. The Fast Fourier Transform performs better, but I am doing from scratch for learning
#   purposes
#   Continuous 'Fourier Transform' :  X(F) = ∫[∞,-∞] x(t) * e^(-j*2*π*F.t) * dt
#   Discrte 'Fourier Transform'    :  Xk = SUM[n=0,N-1] xn  * e^((-j*2*π*k*n)/N) * dt
#      k/N ~ F 
#      n ~ t
def DiscreteFourierTransform(values):
    print("DiscreteFourierTransform")
    numberOfSamples = values.size
    NegativePiDivBySamples = -math.pi*2/numberOfSamples

    returnObject = np.array([0 + 0j] * values.size) #declarying an array of complex numbers

    sumReal = 0.0
    sumImag = 0.0
    complexNumber = 0 + 0j

    for k in range(0,numberOfSamples):
        # print("k: ",k)
        for n in range(0,numberOfSamples):
            # print("    n: ",n)
            #we use value at index n
            value = values[n]
            expValue = NegativePiDivBySamples * k * n
            # Euler's Formula: e^(j*x) = cos x + j * sin x
            _cos = math.cos(expValue)
            _sin = math.sin(expValue)
            valTimesCos = value * _cos
            valTimesSin = value * _sin
            sumReal += valTimesCos
            sumImag += valTimesSin
        complexNumber = complex(sumReal,sumImag)
        returnObject[k] = complexNumber
        sumReal = 0
        sumImag = 0
        
    return returnObject



def FullFourierTransformWithGraph():
    print("Discrete FT")
    #-----
    # Set up input data

    # FYI: in case you see me multiplying numbers below, it's just because I am experimenting; either trying to extend the time, increase the sampling points, etc
    timeSpan = np.linspace(0, 0.875*5, 8*5*32) #from 0 seconds, to 0.875 seconds, with 8 sampling points

    # timeSpan = np.linspace(0, 3.1, 32) # just testing

    values = (
          np.cos(     2 * np.pi * timeSpan - 1.571) 
        + np.cos(13 * 2 * np.pi * timeSpan - 1.571)
        + np.cos(7  * 2 * np.pi * timeSpan - 1.571)
        + np.cos(15 * 2 * np.pi * timeSpan - 1.571)
    )

    # values = np.cos(2 * np.pi * timeSpan) # just testing

    #-----
    # Set up internal variables
    numberOfSamples = values.size
    stepInterval = timeSpan[1] - timeSpan[0] # sampling STEP interval 
    deltaFreq = 1/ (numberOfSamples * stepInterval)
    oneDivByStepInterval = 1 / stepInterval

    frequencies = np.linspace(0, oneDivByStepInterval, numberOfSamples) # this basically is python's use for deltaFreq
    X_UsableFrequencies = frequencies[0:numberOfSamples//2] #takes the half of the frequencies available - Nyquist frequency cutoff

    #-----

    # Fourier Transform - In this case Fast Fourier Transform
    fft = DiscreteFourierTransform(values)
    # fft = np.fft.fft(values)

    # Absolute value SQRT( a^2 + bi^2 )
    # npabs_fft_ = np.abs(fft)  
    FF_Absolute = AbsoluteValueForComplexNumbers(fft) #made from scratch just for learning purposes
      
    

    # Magnitude - Adjusting scale, we only use X/2 of the samples because of Nyquist frequency cutoff
    #  to compensate for that we multiply the value for 2 and then 1/N to make magnitude = 1
    #  e.g.: A signal of 1Hz with 8 samples, will generate 3.9997 magnitude at 1Hz
    #        we then multiply it by 2 (Nyquist cutoff) => 3.9997*2 = 7.9994
    #        and then divide by the number of samples => 7.9994/8 = 0.999925 ~~~~> 1.0
    Y_Amplitude = FF_Absolute[0:numberOfSamples//2] * 2 * 1 / numberOfSamples 

    #-----
    # Show a graph representing the frequency over time
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.plot(timeSpan, values)
    plt.show()
    #-----

    #-----
    # Frequency vs Magnitude graph
    plt.bar(X_UsableFrequencies,
            Y_Amplitude,    
            width=1)

    plt.show()        
    #-----

    debugBreakPointWait = 0

def main():
    FullFourierTransformWithGraph()

if __name__ == "__main__":
    main()