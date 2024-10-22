dt=$(date '+%Y%m%d-%H%M%S');
#echo "$dt"
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python -m adventuregpt -o "../adventuregpt-output-$dt.txt"