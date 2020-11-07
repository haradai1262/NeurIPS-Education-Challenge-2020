# Task 3: Global Question Quality Assessment

## Our Solution
For devising suitable metrics to measure the quality of the diagnostic questions, we formed a hypothesis that an appropriate diagnostic question strikes
  - a balance between the choice of answers
  - an appropriate level of difficulty
  - readability

Based on this hypothesis, we compute below features.
  1. Selection entropy
   - Utilize the variation of `AnsweValue`
  2. Correct/Wrong/entropy
   - Utilize the variation of `IsCorrect`
  3. Difficulty
   - Compute the difference between the mean correctness rate of a student who answered a question and whether the student’s answer to the question is correct or wrong
  4. Readability
   - Extract text regions from a question image and then calculated the proportion of text area to whole area of the image utilizing [CRAFT](https://arxiv.org/pdf/1904.01941.pdf)

## How to Use
1. Utilize CRAFT to each question image
  See the detail in https://github.com/clovaai/CRAFT-pytorch
2. Put the result files into `../data/images_text-segmentation`
3. Run `task3-solution.ipynb`
