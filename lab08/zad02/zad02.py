from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Pobranie zasobów Vader
nltk.download('vader_lexicon')

# Inicjalizacja SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Opinie
review_1 = ('Location was brilliant in a relatively safe area and just a couple of minutes away from Earl’s Court tube '
            'station. So many amenities, food and drinks options nearby. Exceptional service by all staff present '
            'from reception, to bar and kitchen, to cleaning staff. I was there ahead of check-in time and had wanted '
            'to leave my bags at reception but the reception staff were so kind to let me check-in early as the room '
            'was ready and was nothing but excellent and wonderful. Everyone had made our stay a very pleasant one.')

review_2 = 'Very very disappointing. The rooms are tiny, no proper heating or air conditioning. I was told the hotel is in a soft launch and they are aware of these teething problems however no one wanted to help me or take ownership I was told once checked out to write an email of complaint which I’m still waiting to hear back. Very said travelled 6 hours to take my mum away for her 55th birthday and I ended up apologising as I wouldn’t have booked this hotel if I was to  know the issue we were faced. We felt unsafe and wedge the door closed with our suitcases as the lock did not work properly and a lady in reception was complaining someone walked into her room and opened her door.'

review_3 = 'You expect for a 5-star hotel excellent service, but unfortunately that is not the case. The first problem was when I was showering there came water off the ceiling and the whole bathroom got wet. Then we complained and went on with our day. When we came back NOTHING was cleaned. Only the bed was made (with the same dirty bedding). Besides that everything was on the floor, the cleaner dropped stuff on our designer items without shame. My laundry was also not picked up and when I wanted to complain, they could do nothing except offer a upgrade the new stay. I won’t be coming here after the horror I been through. The only thing I want is to get my money back for the poor 2 days I stayed. I don’t understand the 5-star reviews honestly.'

review_4 = 'It was the worst hotel I have ever stayed in. The lobby smelled strongly of food. As soon as I entered the room, I saw that it had nothing to do with the promotional photos. The dirty carpet, the glass cups full of lip marks made me sick. There were hairs everywhere in the bathroom. The faucet was broken and there was no hot water. It was in such a terrible condition that I was disgusted to take a shower. I tried to reach the reception and never got a response. The QR code they left in the room did not work. Even though I complained to the reception the next day, when I returned to the room in the evening, everything was still very dirty. Also, the hotel never felt safe. Every time I passed through the lobby, there were customers constantly complaining and trying to check out. I do not recommend anyone to stay here. It was a nightmare. I also recorded it on video but it cannot be shared here.'

review_5 = 'I only booked this for one night but it was a heavenly experience. The front desk staff were wonderfully welcoming and even upgraded my room! My sleep was so comfy and cosy on the double bed and even though it had been a late night for a birthday party, I awoke feeling refreshed. The breakfast was also great, streaky bacon, properly done scrambled eggs and cute mini condiments. The shower was bliss and the complimentary creams, shampoos, robes, simply wonderful. I am going to stay here again as my friend lives nearby in Kenny.'
# Analiza sentymentu dla obu opinii
scores_review_1 = sia.polarity_scores(review_1)
scores_review_2 = sia.polarity_scores(review_2)
scores_review_3 = sia.polarity_scores(review_3)
scores_review_4 = sia.polarity_scores(review_4)
scores_review_5 = sia.polarity_scores(review_5)
# Wyświetlenie wyników
print("Review 1 Sentiment Scores:", scores_review_1)
print("Review 2 Sentiment Scores:", scores_review_2)
print("Review 3 Sentiment Scores:", scores_review_3)
print("Review 4 Sentiment Scores:", scores_review_4)
print("Review 5 Sentiment Scores:", scores_review_5)

'''
Review 1 Sentiment Scores: {'neg': 0.047, 'neu': 0.736, 'pos': 0.217, 'compound': 0.9661}
Review 2 Sentiment Scores: {'neg': 0.112, 'neu': 0.859, 'pos': 0.029, 'compound': -0.8742}
Review 3 Sentiment Scores: {'neg': 0.17, 'neu': 0.745, 'pos': 0.085, 'compound': -0.9461}
Review 4 Sentiment Scores: {'neg': 0.148, 'neu': 0.824, 'pos': 0.027, 'compound': -0.9506}
Review 5 Sentiment Scores: {'neg': 0.0, 'neu': 0.615, 'pos': 0.385, 'compound': 0.9946}
'''

# d) Czy wyniki testów są zgodne z oczekiwaniami?
# Tak, wyniki testów są zgodne z oczekiwaniami. Opinie pozytywne mają wyższy poziom emocji "pos", podczas gdy negatywne mają wyższy poziom emocji "neg".