#Tracking the tracker: Time Series Analysis in Python From First Principles

This is a source code that accompany the PyCon Canada 2017 talk which was held in November 17 - 21.

The source code are original work of the author, so any resemblance to existing work on the Internet would be merely coincidental. This work is free from every form of plagarism, so the references of the research papers used in writing the codes are provided.

Why would the author spend time implementing from first principle or reinventing the wheel. This was done to ensure that the audience internalize the concept described as the implementation are free of unneccessary optimizations that destroy the flow of the talk.

#Run the Test


#API
See the examples folder.


#Q & A

is the code production ready?
You may want to offload aspects of the processing to Cython. Otherwise, it can be suitable for your needs. This is an experimental project.

is the code free from bugs?
This code does not have any intentional bug. However, like Donald Knuth would say 'Beware of bugs in the above code; I have only proved it correct, not tried it.'. Only one caveat here, I have demonstrated a use case to show the usefulness of the code.


Feel free to reach out to me for any discussion about anything.
Twitter:  @kenluck2001
Linkedin: https://www.linkedin.com/in/kenluck2001/


#References

1. Castanon, D., & Karl, C. W. SC505: Stochastic processes. Retrieved 06/15, 2017, from http://www.mit.edu/people/hmsallum/GradSchool/sc505notes.pdf

2. Wellstead, P. E. & Karl, C. W. (1991). Self-tuning Systems: Control and Signal Processing. Chichester, United Kingdom: John Wiley & Sons Ltd.

3. Hamilton, J. D. (1994). Time series analysis. Chichester, United Kingdom: Princeton University Press.

4. Julier, S. J. The scaled unscented transformation. Retrieved 06/15, 2017, from https://www.cs.unc.edu/~welch/kalman/media/pdf/ACC02-IEEE1357.PDF

5. Terejanu, G. A. Extended kalman filter tutorial. Retrieved 06/15, 2017, from https://www.cse.sc.edu/~terejanu/files/tutorialEKF.pdf

6. Wan , E. A., & Merwe, R. (2000). The unscented kalman filters for nonlinear estimation. IEEE Adaptive Systems for Signal Processing, Communications, and Control Symposium, pp. 153-158.

7. Welch, G., & Bishop, G. An introduction to the kalman filter. Retrieved 06/15, 2017, from https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf 
