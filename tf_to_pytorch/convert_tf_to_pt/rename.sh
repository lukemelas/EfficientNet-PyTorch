for i in 0 1 2 3 4 5 6 7 8
do 
    X=$(sha256sum efficientnet-b${i}.pth | head -c 8)
    mv efficientnet-b${i}.pth efficientnet-b${i}-${X}.pth
done
