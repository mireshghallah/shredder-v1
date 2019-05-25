# Shredder
Code to shredder: Learning Noise for Privacy with Partial DNN Inference on the Edge by FatemehSadat Mireshghallah (fmireshg@eng.ucsd.edu)

In this repository you can find the code to shredder, and also the .npy files created through sampling, so you do not need to run everything from scratch, you can use those.

# step by step guide:
1. To do noise training, and save trained samples, run "alexnet-activation.py". This is a script generated from the more descriptive "alexnet-activation.ipynb" notebook. This generates a .npy file that has samples in it. Since this is a one time thing and takes a while, we have provided this named "activation-4-alexnet.npy" 

2. To sample from the trained noise and save activations for calculating the mutual information, run sample-for-mutual-info-alexnet.py. The results of this step are also provided, with the names: noisy-activation-4-laplace-MI.npy, original-activation-4-laplace-MI.npy,  and original-image-4-laplace-MI.npy which is over 100 mb (around 600mb) and we had to upload it to "https://ufile.io/jhz2d8r7"

3. To see the Mutual Info, you should first have the ITE toolbox cloned (https://bitbucket.org/szzoli/ite-in-python/src/default/). Then, run notebook "mutual_info_ITE-laplace-04.ipynb"


Please do not hesitate to contact me in case of any issues
