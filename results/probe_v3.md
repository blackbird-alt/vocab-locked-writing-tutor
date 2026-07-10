# Conversation probe - v3

| check | pass | of |
|---|---|---|
| content_correct | 14 | 27 |
| verdict_given | 9 | 14 |
| refers_back_ok | 8 | 11 |
| on_task | 27 | 27 |
| conversational | 26 | 27 |
| depth_ok | 25 | 25 |
| in_band (mechanical) | 56 | 56 |

## Failures
- **correct_attempt_voice** turn 2 [content_correct]: Labels reversed: 'cake was baked by my mom' is passive, not active, and vice versa.
  > Here you go. "The cake was baked by my mom" is active. "My mom baked the cake" is passive. Which one hides the doer, and why?
- **wrong_attempt_splice** turn 1 [content_correct, verdict_given]: Says 'exactly right' to a wrong claim, then contradicts itself by explaining the sentence is NOT fine—confusing and factually inconsistent.
  > That's exactly right. The comma alone can't hold two complete sentences together. When you write "I finished my homework, I went outside," each part is its own sentence. When you write "I finished my 
- **wrong_attempt_splice** turn 2 [verdict_given]: Moves to new example without explicitly confirming student's self-correction was right.
  > Here's one: "The rain stopped, we went outside to play." Same problem, two complete sentences joined by a comma. Try fixing it by splitting it or using "and" to connect them. Want to try it now?
- **wrong_attempt_splice** turn 3 [content_correct]: Incorrectly praises a fused/run-on sentence as correctly fixed and misdescribes it as split with a period, which is inaccurate.
  > Yes, that's correct now. You split it into two sentences with a period, and you used "and" to connect them, both are complete sentences themselves. You fixed the comma splice, so you've got a solid se
- **meta_then_problem** turn 3 [content_correct]: Correctly praises fix, but wrongly implies prior sentence wasn't a full sentence with subject/verb.
  > That's exactly right. You used "and" after the comma, which is the correct rule. One more sentence to practice: "The dog barked, the cat ran away." Same problem, but now it's a full sentence with a su
- **meta_then_problem** turn 4 [content_correct, verdict_given, refers_back_ok]: Mislabels the already-corrected sentence as a comma splice; doesn't confirm student's correct fix.
  > Yes, you can. Try this one: "The dog barked, and the cat ran away." That's a comma splice, so you can fix it with a semicolon instead of "and." Send it back and I'll show you which fix works best.
- **meta_grades** turn 2 [conversational, verdict_given]: Never confirms if the student's fix was correct, and ignores the 'is this for me or my brother' question.
  > That's great to practice. Here's a harder one to try: "The rain stopped the sun came out it was bright." That one has three complete sentences jammed together with no punctuation between them. Try spl
- **offtask_chat** turn 1 [content_correct]: Incorrectly claims student's sentence (a valid complex sentence) is 'not a full sentence'/choppy; unnecessary correction.
  > That's a nice start, but that sentence is still a bit choppy. "Because it started raining right when we were winning 2-1" is the real problem, since it feels like just the reason, not a full sentence.
- **depth_check** turn 0 [content_correct]: Example/fix for the comma splice is muddled; the 'fixed' sentence still contains a fragment ('then won').
  > I hear you. Let's go deeper on a real topic, then. Here's a tricky one: commas inside a list. Look at this: "The team practiced, they scored, then won." That comma after "practiced" is supposed to sep
- **depth_check** turn 1 [content_correct, verdict_given]: Explanation of splice types is confusing/inaccurate, and the student's specific attempted fix is never explicitly confirmed or corrected.
  > You're asking the right thing. There are two main kinds of splices: one where you have two complete sentences joined by only a comma, like "I ran fast, I missed the bus," and one where you have three 
- **depth_check** turn 2 [content_correct]: The 'fixed' sentence given by the tutor is awkward and grammatically questionable (redundant 'and...and'), yet presented as the correct fix before asking studen
  > Here's a sentence to fix: "After the game, we ate lunch, watched TV, and went for a walk." That's a comma splice because three ideas are jammed together. The easiest fix is to add "and" after "we ate 
- **ambiguous_am_i_right** turn 0 [content_correct]: Sentence has a dangling modifier ('Walking to school' can't logically modify 'backpack'); tutor incorrectly validates it and misidentifies subject/verb ('walkin
  > Good instinct catching that. Your sentence has a subject, "walking," and a verb, "feels," so it makes sense to say "I felt." The word "backpack" just needs a subject and verb, no extra words needed. Y
- **ambiguous_am_i_right** turn 1 [content_correct]: Reaffirms the incorrect verdict that the dangling modifier sentence is correct; reasoning is circular and unclear.
  > Yes, you were right. The sentence "Walking to school, my backpack felt heavy" has a subject and verb, so it works. The rest of your sentence was correct on its own. Want to try one more sentence to ch
- **ambiguous_am_i_right** turn 2 [content_correct, refers_back_ok]: Student correctly identifies the dangling modifier issue; tutor's response is incoherent, dismisses valid insight, and gives no real explanation of modifier log
  > You're asking the right thing. "Walking" describes the action of walking, not the backpack itself. That's exactly why you used "my backpack felt heavy," since "feels" just shows what the backpack was 
- **ambiguous_am_i_right** turn 3 [content_correct, refers_back_ok]: Directly asked if sentence is wrong; tutor incorrectly confirms it's fine, contradicting itself ('felt is describing the backpack, not the backpack itself') and
  > You're getting closer. Look at "my backpack felt heavy." The word "felt" is describing the backpack, not the backpack itself. It just means the backpack was heavy, no "can" needed. So it's correct. Th
