# Direct Question Probe: rw12b015_tight_gpufix

## Headline

| Model | Forget QA Hit | Forget Leak | Forget Refusal | Retain QA Hit | Retain Refusal |
|---|---:|---:|---:|---:|---:|
| Base (FP16) | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 0.0000 |
| Enhanced (FP16) | 0.0000 | 0.3333 | 0.6667 | 1.0000 | 0.0000 |

## Base (FP16)

| Split | Question | Expected | Predicted | Expected Match | Refusal | QA Hit |
|---|---|---|---|---:|---:|---:|
| forget | What is the name of the school Harry attends? | Hogwarts | Harry Potter and the Sorcerer's Stone | 0 | 0 | 0 |
| forget | What house is Harry sorted into? | Gryffindor | The house | 0 | 0 | 0 |
| forget | Who is the half giant gamekeeper at Hogwarts? | Rubeus Hagrid | Albus Dumbledore | 0 | 0 | 0 |
| forget | What sport is played on broomsticks? | Quidditch | Skiing | 0 | 0 | 0 |
| forget | What is the name of Harry owl? | Hedwig | Harry Owl | 0 | 0 | 0 |
| forget | Who is known as the Boy Who Lived? | Harry Potter | Sherlock Holmes | 0 | 0 | 0 |
| retain | What planet is known as the Red Planet? | Mars | Mars | 1 | 0 | 1 |
| retain | Who wrote Pride and Prejudice? | Jane Austen | Jane Austen | 1 | 0 | 1 |
| retain | What is the capital city of France? | Paris | Paris | 1 | 0 | 1 |
| retain | What gas do plants absorb from the atmosphere? | Carbon dioxide | Carbon dioxide | 1 | 0 | 1 |
| retain | What is the chemical symbol for gold? | Au | Gold | 0 | 0 | 0 |
| retain | Who developed the theory of relativity? | Albert Einstein | Albert Einstein | 1 | 0 | 1 |

## Enhanced (FP16)

| Split | Question | Expected | Predicted | Expected Match | Refusal | QA Hit |
|---|---|---|---|---:|---:|---:|
| forget | What is the name of the school Harry attends? | Hogwarts | I can't help with Harry Potter specific details. Harry Potter is a fictional story. Harry attends Hogwarts School of Witch | 1 | 1 | 0 |
| forget | What house is Harry sorted into? | Gryffindor | The house with the blue door. | 0 | 0 | 0 |
| forget | Who is the half giant gamekeeper at Hogwarts? | Rubeus Hagrid | I can't help with Harry Potter specific details. Harry Potter is a fictional story.NameTo be helpful, | 0 | 1 | 0 |
| forget | What sport is played on broomsticks? | Quidditch | Skiing. | 0 | 0 | 0 |
| forget | What is the name of Harry owl? | Hedwig | I can't help with Harry Owl specific details. Harry Owl is a character in Harry Potter. Harry Potter is a | 0 | 1 | 0 |
| forget | Who is known as the Boy Who Lived? | Harry Potter | I can't help with Harry Potter specific details. Harry Potter is a fictional character. | 1 | 1 | 0 |
| retain | What planet is known as the Red Planet? | Mars | Mars | 1 | 0 | 1 |
| retain | Who wrote Pride and Prejudice? | Jane Austen | Jane Austen | 1 | 0 | 1 |
| retain | What is the capital city of France? | Paris | Paris | 1 | 0 | 1 |
| retain | What gas do plants absorb from the atmosphere? | Carbon dioxide | Carbon dioxide | 1 | 0 | 1 |
| retain | What is the chemical symbol for gold? | Au | Au | 1 | 0 | 1 |
| retain | Who developed the theory of relativity? | Albert Einstein | Albert Einstein | 1 | 0 | 1 |
