curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1KkXffb1DEu1be7NO-RPUy1r2bZqJRuYl" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1KkXffb1DEu1be7NO-RPUy1r2bZqJRuYl" -o data.zip

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1IlHzuFeAMbPzxLCghaFzDV1FPuXwwcC0" -o snapshots.zip

mkdir data
mkdir snapshots
unzip data.zip -d data
unzip snapshots.zip -d snapshots
rm data.zip
rm snapshots.zip
