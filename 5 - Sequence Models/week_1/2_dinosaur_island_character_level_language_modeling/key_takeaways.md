## What you should remember:
* Very large, or "exploding" gradients updates can be so large that they "overshoot" the optimal values during back prop -- making training difficult 
    * Clip gradients before updating the parameters to avoid exploding gradients 
* Sampling is a technique you can use to pick the index of the next character according to a probability distribution.
    * To begin character-level sampling: 
        * Input a "dummy" vector of zeros as a default input 
        * Run one step of forward propagation to get 𝑎⟨1⟩ (your first character) and 𝑦̂ ⟨1⟩ (probability distribution for the following character) 
        * When sampling, avoid generating the same result each time given the starting letter (and make your names more interesting!) by using `np.random.choice`