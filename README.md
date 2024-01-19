# Text-to-SQL-Converter-with-Transformer-Architecture
Code to implemnent a transformer model that converts natural text in to SQL queries  

## Overview

This code implements a complete transformer model (encoder + decoder) with a multihead attention mechanism. This model was trained using the "Clinton Text-to-SQL v1." dataset and produces a generative SQL query that corresponds to the natural text query, with SQL context, that was input. 

## Project Structure 

1. tokeniser.py - creates and trains sentencepiece tokeniser
  
2. dataset.py -  implements tokenised dataset
   
3. decoder.py - implements transformer architecture 

4. train.py - trains the transformer using Clinton Text-to-SQL v1 dataset

5. inference.py - generates output SQL query from input text and context 

6. constants.py - lists all the constants used across the modules 
    
## Author 

Louis Chapo-Saunders
