------------------------------------------------------------------------
--[[ Recursor ]]--
-- Decorates module to be used within an AbstractSequencer.
-- It does this by making the decorated module conform to the 
-- AbstractRecurrent interface (which is inherited by LSTM/Recurrent) 
-- adapted from Recursor.lua (11/13/2016) by mhahn
------------------------------------------------------------------------
require 'nn'

local RecursorMod, parent = torch.class('nn.RecursorMod', 'nn.Recursor')

-- encode for each output whether it is fed back
-- wiring: for each output, say true if it should be fed back into the recurrent module
function RecursorMod:__init(module, rho, initialValuesOfRecurrentInputs, numberOfExternalInputs, numberOfOutputs, wiring)
   parent.__init(self, module, rho or 9999999)

   self.numberOfOutputs = numberOfOutputs
   self.numberOfExternalInputs = numberOfExternalInputs
   assert(#wiring == self.numberOfOutputs)
   self.recurrentWiring = wiring
   self.initialValuesOfRecurrentInputs = initialValuesOfRecurrentInputs

   self.gradInputs_per_steps = {}
   self.gradOutputInternal_per_steps = {}

--[[   self.recurrentModule = module
   
   self.module = module
   self.modules = {module} -- this is important for collecting the parameters
   self.sharedClones[1] = self.recurrentModule]]
end

function RecursorMod:assembleInputToModule(step,input)
      local inputToModule = {}
      for i=1,#input do
        table.insert(inputToModule,input[i])
      end
      if step == 1 then
        for i=1,#self.initialValuesOfRecurrentInputs do
          table.insert(inputToModule, self.initialValuesOfRecurrentInputs[i])
        end
      else
        for i=1,#(self.outputs[step-1]) do
          if(self.recurrentWiring[i] == true) then
             table.insert(inputToModule, self.outputs[step-1][i])
          end
        end
      end
      assert(#inputToModule == self.numberOfExternalInputs + #self.initialValuesOfRecurrentInputs)
      return inputToModule
end




-- note that the inputs and outputs of the recurrent module are assumed to be tables
-- in the current state, there is no backpropagation into the initial states of the recurrent inputs (maybe they should become inputs to sequencer)
function RecursorMod:updateOutput(input)
--   print(input)
   assert(#input == self.numberOfExternalInputs)
   local output
   if self.train ~= false then -- if self.train or self.train == nil then
      -- set/save the output states
      self:recycle()
      local recurrentModule = self:getStepModule(self.step)
      local inputToModule = self:assembleInputToModule(self.step, input)
      output = recurrentModule:updateOutput(inputToModule)
--[[      print(": "..self.step)
      print(input)
      print(inputToModule)
      print(inputToModule[1][1])
      print(inputToModule[2][1][1])
      print(inputToModule[3][1][1])

      print(output[1][1][1])
      print(output[2][1][1])
      print(output[3][1][1])]]

   else
      crash()
      output = self.recurrentModule:updateOutput(input)
   end
   assert(#output == self.numberOfOutputs)
   self.outputs[self.step] = output
   self.output = output
   self.step = self.step + 1
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   return self.output
end

-- note that this so far is called twice, once by updateGradInput and once by accGradParameters, which is a bit wasteful
function RecursorMod:assembleGradOutputInternal(step,gradOutput)
  assert(step>=1)
  local gradOutputInternal = self.gradOutputInternal_per_steps[step]
  if gradOutputInternal == nil then
     self.gradOutputInternal_per_steps[step] = {}
     gradOutputInternal = self.gradOutputInternal_per_steps[step]
  end

  -- get gradients coming from the output
  -- note that it is necessary to clone, since later the gradients from the recurrent connections will be added
  for i=1,#gradOutput do
    if gradOutputInternal[i] == nil then
      table.insert(gradOutputInternal, gradOutput[i]:clone())
    else
      gradOutputInternal[i]:copy(gradOutput[i])
    end
  end

  -- get gradients coming from the recurrent connections
  if self.gradInputs_per_steps[step+1] ~= nil then
    local recurrentConnectionsSoFar = 0
    for i=1,#self.recurrentWiring do
      if self.recurrentWiring[i] == true then
        recurrentConnectionsSoFar = recurrentConnectionsSoFar + 1
        gradOutputInternal[i]:add(self.gradInputs_per_steps[step+1][self.numberOfExternalInputs+recurrentConnectionsSoFar])
      end
    end
    assert(self.numberOfExternalInputs+recurrentConnectionsSoFar == #(self.gradInputs_per_steps[step+1])) 
  end

  assert(#gradOutput == #gradOutputInternal)
  
  return gradOutputInternal
end


function RecursorMod:_updateGradInput(input, gradOutput)
   assert(self.step > 1, "expecting at least one updateOutput")
   local step = self.updateGradInputStep - 1
   assert(step >= 1)
  
   local inputToModule = self:assembleInputToModule(step, input)

   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)

   local gradOutputInternal = self:assembleGradOutputInternal(step, gradOutput)
   

   local gradInput = recurrentModule:updateGradInput(inputToModule, gradOutputInternal)
   
   local gradExternalInput = {}
   for i=1,#input do
     gradExternalInput[i] = gradInput[i]
   end
   self.gradInputs_per_steps[step] = gradInput
--[[   for i=1,#self.recurrentWiring do
     if self.recurrentWiring[i] then
       table.insert(self.recurrentGradInput[step], gradInput[i + #input])
     else
   end
   assert(#(self.recurrentGradInput[step]) == #(self.initialValuesOfRecurrentInputs))]]
   assert(#input == #gradExternalInput)
   return gradExternalInput
end

function RecursorMod:_accGradParameters(input, gradOutput, scale)
   local step = self.accGradParametersStep - 1
   assert(step >= 1)
 
   local inputToModule = self:assembleInputToModule(step, input)
assert(self.gradOutputInternal_per_steps[step])

   local gradOutputInternal =  self.gradOutputInternal_per_steps[step]
--    self:assembleGradOutputInternal(step, gradOutput)


   local recurrentModule = self:getStepModule(step)
   recurrentModule:setOutputStep(step)
   recurrentModule:accGradParameters(inputToModule, gradOutputInternal, scale)
end

--[[function RecursorMod:includingSharedClones(f)
   local modules = self.modules
   self.modules = {}
   local sharedClones = self.sharedClones
   self.sharedClones = nil
   for i,modules in ipairs{modules, sharedClones} do
      for j, module in pairs(modules) do
         table.insert(self.modules, module)
      end
   end
   local r = {f()}
   self.modules = modules
   self.sharedClones = sharedClones
   return unpack(r)
end

function RecursorMod:forget(offset)
   parent.forget(self, offset)
   nn.Module.forget(self)
   return self
end

function RecursorMod:maxBPTTstep(rho)
   self.rho = rho
   nn.Module.maxBPTTstep(self, rho)
end]]

RecursorMod.__tostring__ = nn.Decorator.__tostring__

