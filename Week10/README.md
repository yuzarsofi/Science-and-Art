# Week 10: AI Parameter Experiments

## Experiment 1: Temperature
I tested the prompt: "Write a creative story about a cat..."

* **Low Temperature (0.1):** The AI was very repetitive. It said: "Attempt 1: A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat.
Attempt 2: A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat.
Attempt 3: A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat. A cat is a cat."

* **High Temperature (0.9):** The AI was more creative. It said: "Attempt 1: The cat is a poodle. It is a cat that likes to play fetch. One day, a poodle runs into the cat's house. The poodle chases after the cat. The p
Attempt 2: The President of the United States is a cat who lives in the United States. He is a cat who is a president of the United States. He is a cat who lives in the United States. He is a cat who
Attempt 3: Cats are a great way to spend a day. They are a great way to spend a day. Cats are a great way to spend a day. Cats are a great way to spend a day"

## Experiment 2: Top_P (Vocabulary Choice)
**Prompt:** "Explain quantum physics to a 5-year-old."

* **Low Top_P (0.5):**
    * *Observation:* The AI used very common, simple words. It felt more focused.
    * *Sample Output:* "Attempt 1: Using a microscope, a student observes a ray of light passing through a small hole in the center of a sphere. The student observes that the ray passes through the hole, and that the sphere
Attempt 2: Using a microscope, a student observes a light bulb in a room. The student observes that the light bulb is emitting ultraviolet light. The student observes that the light bulb is emitting ultraviolet light. The student
Attempt 3: Using a microscope, a student observes a ray of light coming through a hole in the wall of a building. The student observes the light coming through the hole. The student explains that the light coming through"

* **High Top_P (0.95):**
    * *Observation:* The AI used a wider variety of words and sentence structures.
    * *Sample Output:* "Attempt 1: Physics is a field of study that focuses on the physical properties of matter. Quantum physics is a field of study that focuses on the physical properties of matter. Quantum physics is a field of study that focuses
Attempt 2: Using a microscope, a student observes that a light bulb is emitting a light. The student uses a microscope to observe the light bulb. The student observes that the light bulb is emitting a light bulb
Attempt 3: Physics is a field of study that focuses on the physical properties of matter. The physical properties of matter can be compared to the physical properties of matter. The physical properties of matter can be compared to the physical properties of matter."

## Conclusion
* **Temperature** controls the "randomness" of the next token. High temp = wild; Low temp = robotic.
* **Top_P** (Nucleus Sampling) limits the word choices to the top percentage. Low Top_P removes rare/surprising words.