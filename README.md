# OSGM

* Python 3.7.x:

# <h3>Example:</h3>

python main.py -d "data/Tweets-T-N" -o "result/" -lcb -stc -icf -cww -decay 0.000001 -mclus -ft 10 -invb -alpha 0.05 -beta 0.003

# <h3>Example#2:</h3>

python main.py -d "data/Tweets-T-N" -o "result/" -lcb -stc -icf -cww -decay 0.000001 -mclus -ft 10 -invb

# <h3>Example#3:</h3>

python main.py -d "data/Tweets-T-N" -o "result/" -lcb -stc -cww -decay 0.000001 -mclus -ft 10 -invb

# <h3>Parameters Definitions:</h3>
* -icf  : include inverse cluster frequency
* lcb   : include cluster-based beta value
* cww   : include word-to-word co-occurrence probability
* mclus : enable merging of outdated clusters
