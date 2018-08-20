------------------------------------------------------------------------
--[[ Sequencer ]]--
-- Encapsulates a Module. 
-- Input is a sequence (a table) of tensors.
-- Output is a sequence (a table) of tensors of the same length.
-- Applies the module to each element in the sequence.
-- Handles both recurrent modules and non-recurrent modules.
-- The sequences in a batch must have the same size.
-- But the sequence length of each batch can vary.
-- modified mhahn 2016
------------------------------------------------------------------------

require 'nn'

assert(not nn.SequencerMod, "update nnx package : luarocks install nnx")
local SequencerMod, parent = torch.class('nn.SequencerMod', 'nn.Sequencer')
local _ = require 'moses'

function SequencerMod:__init(module)
   parent.__init(self,module)
--[[   if not torch.isTypeOf(module, 'nn.Module') then
      error"SequencerMod: expecting nn.Module instance at arg 1"
   end
   
   -- we can decorate the module with a Recursor to make it AbstractRecurrent
   self.module = (not torch.isTypeOf(module, 'nn.AbstractRecurrent')) and nn.Recursor(module) or module
   -- backprop through time (BPTT) will be done online (in reverse order of forward)
   self.modules = {self.module}
   
   self.output = {}
   self.tableoutput = {}
   self.tablegradInput = {}
   
   -- table of buffers used for evaluation
   self._output = {}
   -- so that these buffers aren't serialized :
   local _ = require 'moses'
   self.dpnn_mediumEmpty = _.clone(self.dpnn_mediumEmpty)
   table.insert(self.dpnn_mediumEmpty, '_output')
   -- default is to forget previous inputs before each forward()
   self._remember = 'neither']]
end

function SequencerMod:updateOutput(input)
   assert(torch.type(input) == 'table')
   local nStep
   if torch.isTensor(input) then
      nStep = input:size(1)
   else
      assert(torch.type(input) == 'table', "expecting input table")
      nStep = #input
   end

   -- Note that the Sequencer hijacks the rho attribute of the rnn
   self.module:maxBPTTstep(nStep)
   if self.train ~= false then 
      -- TRAINING
      if not (self._remember == 'train' or self._remember == 'both') then
         self.module:forget()
      end
      
      self.tableoutput = {}
      for step=1,nStep do
         self.tableoutput[step] = self.module:updateOutput(input[step])
      end
      
      if torch.isTensor(input) then
         crash()
         self.output = torch.isTensor(self.output) and self.output or self.tableoutput[1].new()
         self.output:resize(nStep, unpack(self.tableoutput[1]:size():totable()))
         for step=1,nStep do
            self.output[step]:copy(self.tableoutput[step])
         end
      else
         self.output = {self.tableoutput, self.tableoutput[nStep]}
      end
   else 
      crash()
      -- EVALUATION
      if not (self._remember == 'eval' or self._remember == 'both') then
         self.module:forget()
      end
      -- during evaluation, recurrent modules reuse memory (i.e. outputs)
      -- so we need to copy each output into our own table or tensor
      if torch.isTensor(input) then
         for step=1,nStep do
            local output = self.module:updateOutput(input[step])
            if step == 1 then
               self.output = torch.isTensor(self.output) and self.output or output.new()
               self.output:resize(nStep, unpack(output:size():totable()))
            end
            self.output[step]:copy(output)
         end
      else
         for step=1,nStep do
            self.tableoutput[step] = nn.rnn.recursiveCopy(
               self.tableoutput[step] or table.remove(self._output, 1), 
               self.module:updateOutput(input[step])
            )
         end
         -- remove extra output tensors (save for later)
         for i=nStep+1,#self.tableoutput do
            table.insert(self._output, self.tableoutput[i])
            self.tableoutput[i] = nil
         end
         self.output = self.tableoutput
      end
   end
   --print("1019")
   --print(self.output) 
   return self.output
end


function SequencerMod:assembleGradOutputs(nStep, gradOutput)
   local gradOutputsPerTimesteps = {}
   for i=1,nStep do
     gradOutputsPerTimesteps[i] = gradOutput[1][i]
   end
   -- can we save some of the copying that is done in the recurrent computations here and in RecursorMod?
   for i=1,#gradOutput[2] do
     -- TODO actually, this should use the nested save that's used in gModule.lua
     gradOutputsPerTimesteps[nStep][i] = torch.add(gradOutputsPerTimesteps[nStep][i], gradOutput[2][i])
   end
--   print("7643")
  -- print(gradOutput)
   --print(gradOutputsPerTimesteps)
   --print(nStep)
   --print("1928")
   --crash()
   return gradOutputsPerTimesteps
end


function SequencerMod:updateGradInput(input, gradOutput)
   --print("4610")
   --print(input)
   --print(gradOutput)

   assert(torch.type(gradOutput) == 'table')
   assert(#gradOutput == 2)
   local nStep
   if torch.isTensor(input) then
      crash()
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      assert(gradOutput:size(1) == input:size(1), "gradOutput should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
      --assert(#gradOutput == #input, "gradOutput should have as many elements as input")
      -- this is not true for this modification
      nStep = #input
   end
   


   local gradOutputsPerTimesteps = self:assembleGradOutputs(nStep, gradOutput)



   -- back-propagate through time
   self.tablegradinput = {}
   for step=nStep,1,-1 do
      self.tablegradinput[step] = self.module:updateGradInput(input[step], gradOutputsPerTimesteps[step])
   end
   
   if torch.isTensor(input) then
      crash()
      self.gradInput = torch.isTensor(self.gradInput) and self.gradInput or self.tablegradinput[1].new()
      self.gradInput:resize(nStep, unpack(self.tablegradinput[1]:size():totable()))
      for step=1,nStep do
         self.gradInput[step]:copy(self.tablegradinput[step])
      end
   else
      self.gradInput = self.tablegradinput
   end
   --print("166")
   --print(self.gradInput)

   return self.gradInput
end

function SequencerMod:accGradParameters(input, gradOutput, scale)
   local nStep
   if torch.isTensor(input) then
      crash()
      assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor since input is a Tensor")
      assert(gradOutput:size(1) == input:size(1), "gradOutput should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
      --assert(#gradOutput == #input, "gradOutput should have as many elements as input")
      --not in the modified version
      nStep = #input
   end
 
   local gradOutputsPerTimesteps = self:assembleGradOutputs(nStep, gradOutput)
  
   -- back-propagate through time 
   for step=nStep,1,-1 do
      self.module:accGradParameters(input[step], gradOutputsPerTimesteps[step], scale)
   end   
end

--[[function SequencerMod:accUpdateGradParameters(inputTable, gradOutputTable, lr)
   error"Not Implemented"  
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
-- Essentially, forget() isn't called on rnn module when remember is on
function SequencerMod:remember(remember)
   self._remember = (remember == nil) and 'both' or remember
   local _ = require 'moses'
   assert(_.contains({'both','eval','train','neither'}, self._remember), 
      "Sequencer : unrecognized value for remember : "..self._remember)
   return self
end

function SequencerMod:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
      -- empty temporary output table
      self._output = {}
      -- empty output table (tensor mem was managed by seq)
      self.tableoutput = nil
   end
   parent.training(self)
end

function SequencerMod:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
      -- empty output table (tensor mem was managed by rnn)
      self.tableoutput = {}
   end
   parent.evaluate(self)
   assert(self.train == false)
end

function SequencerMod:clearState()
   if torch.isTensor(self.output) then
      self.output:set()
      self.gradInput:set()
   else
      self.output = {}
      self.gradInput = {}
   end
   self._output = {}
   self.tableoutput = {}
   self.tablegradinput = {}
   self.module:clearState()
end]]

SequencerMod.__tostring__ = nn.Decorator.__tostring__
