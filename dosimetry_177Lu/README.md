In this repository, the developed code for the **dosimetry** of 177Lu patients is presented. 

- This code was created as part of my masters thesis, where patients treated with 177Lu (specifically PSMA) were analyzed;
- The worflow is as folows:

  <img width="937" height="179" alt="image" src="https://github.com/user-attachments/assets/262aedd5-0ec2-47f8-b491-53d244602bc0" />


- Two sets of data (real and simulated) are studied, to ultimately be compared;
- The entire workflow is run in the *run_main.ipynb* file

About the code:

- The volumes (real and simulated) need to be aligned, as well as the segmentation;
- 5 time-points are needed, with one of them being the (0,0), i.e. the activity in the VOI at injection-time is zero. If less are given, the script does not run and an error is presented;
- The generated absorbed dose maps are normalized to the injected activity. If one wishes to override this, the administred activity value can be set to 1.
The generated Y-Scale in the TACs will correspond to absolute values of activity (MBq/100).
- The Dose Voxel Kernel (VDK) is generated with S-Values obtained for 2.21mm voxel size, 177Lu and Soft-Tissue;
- The TIA maps and their respective resampling to a 2.21mm voxel are saved;
- Consequentely, the generated absorbed dose maps also have 2.21mm cubic voxels. 
  
