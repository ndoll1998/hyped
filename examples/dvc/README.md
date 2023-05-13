# dvc

hyped can be easily integrated into dvc pipelines. This example shows a *simple* and an *advanced* approach.

### simple approach

A *simple* pipeline consists of the three stages, namely data processing, training and evaluating the trained model. This is implemented by `simple/dvc.yaml` for the imdb text classification model.

```
        +---------+      
        | prepare |      
        +---------+      
         *         **    
       **            *   
      *               ** 
+-------+               *
| train |             ** 
+-------+            *   
         *         **    
          **     **      
            *   *        
        +----------+     
        | evaluate |     
        +----------+ 
```

### advanced approach

While the simple *approach* combines the preparation of all data splits in a single stage, a more advanced approach separates data preparation into different stages. The `advanced/dvc.yaml` pipeline implements the same behavior as the *simple* approach following the *advanced* idea. 

```
+---------------+         +-------------+           
| prepare-train |         | prepare-val |           
+---------------+         +-------------+           
              ***         ***                       
                 *       *                          
                  **   **                           
                +-------+         +--------------+  
                | train |         | prepare-test |  
                +-------+*        +--------------+  
                          **         **             
                            **     **               
                              *   *                 
                          +----------+              
                          | evaluate |              
                          +----------+  
```

