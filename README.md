# DCASE 2023 Task 6b 
### Language-Based Audio Retrieval with Distributed Data Parallel
* This subtask is concerned with retrieving audio signals using their sound content textual descriptions (i.e., audio captions). Human written audio captions will be used as text queries. For each text query, the goal of this task is to retrieve 10 audio files from a given dataset and sort them based their match with the query. Through this subtask, we aim to inspire further research into language-based audio retrieval with unconstrained textual descriptions.


```
  Audio Retrieval
  ├── data
  │   ├── csv_files  
  │   ├── waveforms
  │       .....
  │
  ├── data_handling
  │   ├── DataLoader.py  
  │      
  ├── models
  │   ├── ASE_model.py  
  │   ├── AudioEncoder.py  
  │   ├── TextEncoder.py  
  │
  ├── settings
  │   ├── settings.yaml
  │    
  ├── tools
  │   ├── config_loader.py  
  │   ├── dataset.py  
  │   ├── file_io.py  
  │   ├── loss.py    
  │   ├── utils.py    
  │
  ├── trainer
  │   ├── trainer.py  
  │  
  ├── train.py
  
  ```