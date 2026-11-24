Central vs FedAvg. On the 12 test users the central MLP scored a pooled macro F1 of 0.9205 and the selected FedAvg model scored 0.8739, a gap of 4.7 points. Central scored higher for 11 of the 12 users. All results use seed 42 and learning rate 0.03.

Candidate choice. On validation users only, 3 local epochs beat 1 (0.8667 vs 0.8077), so I froze 3 epochs before scoring any test user.

Work done. Central used 49 epochs (39,641 optimizer steps, about 2.54 million examples). FedAvg used 46 rounds of 10 clients (31,740 steps, about 1.98 million examples). These are different kinds of computation, so the table reports cost and does not show which method is better per unit of work.

Client differences. Clients have almost the same size (1,410 to 1,440 windows) and class mix (average distance from the pooled mix 0.0013), so label skew does not explain the gap. People do differ in how they move: for LAccX_mean, client means vary by 0.35 standard deviations against 0.88 within a person.

Mixing control. With seed 42, dealing the training windows into pseudo-clients of the original sizes raised test pooled macro F1 from 0.8739 to 0.8868 (+0.0129). On validation users over 5 seeds, mixed FedAvg beat natural FedAvg at the last round in every seed (differences 0.010 to 0.015, against a seed-to-seed spread of about 0.003 for natural). So the effect is small but consistent. It closes about a quarter of the gap to central, and most of the gap remains. Mixing changed several things at once (it removed the person-level feature differences and made the class mixes slightly uneven), so it cannot show which one matters.

Limits. One main seed for the test comparison, 12 test users, one learning rate, and an untuned 50-round cap.