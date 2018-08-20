binning = {}
binning.number_of_bins = 1 --20
if params.TASK == "combined" then
  binning.number_of_bins = 1
  binning.onlyUseTheFirstNBins = 1
else
  binning.number_of_bins = 100
  binning.onlyUseTheFirstNBins = 99 --20
  print("NOTE: Maybe only using a few bins.")
  print("Only using bins: "..binning.onlyUseTheFirstNBins)
end
binning.do_binning = true

if (DOING_EVALUATION_OUTPUT and CREATE_RECORDED_FILES) or DATASET == 18 then
   print("WARNING: Automatically set binning to FALSE")
   print("Note that binning throws away the end of the dataset, so don't do it when running on the dev/test sets")
   binning.do_binning = false
end

binning.RANDOMIZE_ORDER_OF_BINS = true

assert((DOING_EVALUATION_OUTPUT and DATASET == 18) or (DOING_EVALUATION_OUTPUT and CREATE_RECORDED_FILES) or binning.do_binning or DOING_DEBUGGING)

print("BINNING:")
print(binning)

binning.reading = {}
binning.reading.lastBinReadOutNumber = binning.number_of_bins
binning.reading.bins = {}

function binning.hasNextBin()
  if(binning.onlyUseTheFirstNBins == nil) then
    return(binning.reading.lastBinReadOutNumber < binning.number_of_bins)
  else
    return(binning.reading.lastBinReadOutNumber < binning.onlyUseTheFirstNBins)
  end
end

function binning.getNext(retrieveFunction,batchIndex,lengthFunction)  
   assert(not(DOING_EVALUATION_OUTPUT and CREATE_RECORDED_FILES), "binning will not remember which batch element comes from which file.")

--  print("15")
  if lengthFunction == nil then
     lengthFunction = (function(x) return (#(x.text)+#(x.question)) end)
  end
  --print(batchIndex)
  if(batchIndex == 1) then
--    print(binning.hasNextBin())
  --  print(binning.hasNextBin())
    if((not (binning.hasNextBin()))) then
     binning.reading.lastBinReadOutNumber = 1

     local newElements = {}
     for elementIndex=1,binning.number_of_bins*params.batch_size do
       table.insert(newElements, retrieveFunction(1)) -- assuming that the argument to retrieveFunction is not actually used
     end

     -- now sort
     -- (note that the padding of the directory at the end of the corpus will be short, so no problem with padding the directory)
     table.sort(newElements, function(x,y) return(lengthFunction(x) < lengthFunction(y) ) end)



     for binIndex=1,binning.number_of_bins do
        local newBin = {}
        binning.reading.bins[binIndex] = newBin
        for innerBatchIndex=1,params.batch_size do
           newBin[innerBatchIndex] = newElements[(binIndex-1) * params.batch_size + innerBatchIndex]
           --print("zuhtgvb")
           --print(binIndex.."  "..innerBatchIndex.."  "..(#(newBin[innerBatchIndex].text)))
        end
     end

     
     if binning.RANDOMIZE_ORDER_OF_BINS then
        auxiliary.shuffleTableInPlace(binning.reading.bins)
     end

    else
      binning.reading.lastBinReadOutNumber = binning.reading.lastBinReadOutNumber+1
    end
   end
   --print("READING FROM BIN "..binning.reading.lastBinReadOutNumber)
   --print(#(binning.reading.bins[binning.reading.lastBinReadOutNumber][batchIndex].question))
   --print(#(binning.reading.bins[binning.reading.lastBinReadOutNumber][batchIndex].text))

--   print(binning.reading.bins)
   return binning.reading.bins[binning.reading.lastBinReadOutNumber][batchIndex]
end
