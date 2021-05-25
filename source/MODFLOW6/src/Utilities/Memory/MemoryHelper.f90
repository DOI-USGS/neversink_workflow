module MemoryHelperModule
  use KindModule, only: I4B
  use ConstantsModule, only: LENMEMPATH, LENMEMSEPARATOR, LENMEMADDRESS, LENVARNAME, LENCOMPONENTNAME
  use SimModule, only: store_error, ustop
  use SimVariablesModule, only: errmsg

  implicit none

  character(len=LENMEMSEPARATOR), parameter :: memPathSeparator = '/' !< used to build up the memory address for the stored variables

contains

  !> @brief returns the path to the memory object
  !!
  !! Returns the path to the location in the memory manager where
  !! the variables for this (sub)component are stored, the 'memoryPath' 
  !!
  !! NB: no need to trim the input parameters
  !<
  function create_mem_path(component, subcomponent) result(memory_path)
    character(len=*), intent(in) :: component               !< name of the solution, model, or exchange
    character(len=*), intent(in), optional :: subcomponent  !< name of the package (optional)
    character(len=LENMEMPATH) :: memory_path                !< the memory path
    
    call mem_check_length(component, LENCOMPONENTNAME, "solution/model/exchange")
    call mem_check_length(subcomponent, LENCOMPONENTNAME, "package")  
    
    if (present(subcomponent)) then
      memory_path = trim(component) // memPathSeparator // trim(subcomponent)    
    else
      memory_path = trim(component)
    end if
    
  end function create_mem_path

  !> @brief returns the address string of the memory object
  !!
  !! Returns the memory address, i.e. the full path plus name of the stored variable
  !!
  !! NB: no need to trim the input parameters
  !<
  function create_mem_address(mem_path, var_name) result(mem_address)
    character(len=*), intent(in) :: mem_path    !< path to the memory object
    character(len=*), intent(in) :: var_name    !< name of the stored variable
    character(len=LENMEMADDRESS) :: mem_address !< full address string to the memory object

    call mem_check_length(mem_path, LENMEMPATH, "memory path")
    call mem_check_length(var_name, LENVARNAME, "variable")

    mem_address = trim(mem_path) // memPathSeparator // trim(var_name) 

  end function create_mem_address  

  !> @brief Split a memory address string into memory path and variable name
  !<
  subroutine split_mem_address(mem_address, mem_path, var_name)
    character(len=*), intent(in) :: mem_address         !< the full memory address string
    character(len=LENMEMPATH), intent(out) :: mem_path  !< the memory path
    character(len=LENVARNAME), intent(out) :: var_name  !< the variable name
    ! local
    integer(I4B) :: idx

    idx = index(mem_address, memPathSeparator, back=.true.)

    ! if no separator, or it's at the end of the string,
    ! the memory address is not valid:
    if(idx < 1 .or. idx == len(mem_address)) then
      write(errmsg, '(*(G0))')                                                  &
        'Fatal error in Memory Manager, cannot split invalid memory address: ', &
         mem_address

      ! -- store error and stop program execution
      call store_error(errmsg)
      call ustop()
    end if

    mem_path = mem_address(:idx-1)
    var_name = mem_address(idx+1:)
    
  end subroutine split_mem_address

  !> @brief Split the memory path into component(s)
  !!
  !! NB: when there is no subcomponent in the path, the 
  !! value for @par subcomponent is set to an empty string.
  !<
  subroutine split_mem_path(mem_path, component, subcomponent)   
    character(len=*), intent(in) :: mem_path                      !< path to the memory object
    character(len=LENCOMPONENTNAME), intent(out) :: component     !< name of the component (solution, model, exchange)
    character(len=LENCOMPONENTNAME), intent(out) :: subcomponent  !< name of the subcomponent (package)
    
    ! local
    integer(I4B) :: idx

    idx = index(mem_path, memPathSeparator, back=.true.)
    ! if the separator is found at the end of the string,
    ! the path is invalid:
    if(idx == len(mem_path)) then
      write(errmsg, '(*(G0))')                                               &
        'Fatal error in Memory Manager, cannot split invalid memory path: ', &
         mem_path

      ! -- store error and stop program execution
      call store_error(errmsg)
      call ustop()
    end if


    if (idx > 0) then
      ! when found:
      component = mem_path(:idx-1)
      subcomponent = mem_path(idx+1:)
    else
      ! when not found, there apparently is no subcomponent:
      component = mem_path
      subcomponent = ''
    end if

  end subroutine split_mem_path

  !> @brief Generic routine to check the length of (parts of) the memory address
  !!
  !! The string will be trimmed before the measurement.
  !!
  !! @warning{if the length exceeds the maximum, a message is recorded 
  !! and the program will be stopped}
  !!
  !! The description should describe the part of the address that is checked
  !! (variable, package, model, solution, exchange name) or the full memory path
  !! itself
  !<
  subroutine mem_check_length(name, max_length, description)
    character(len=*), intent(in) :: name        !< string to be checked
    integer(I4B), intent(in)     :: max_length  !< maximum length
    character(len=*), intent(in) :: description !< a descriptive string
    
    if(len(trim(name)) > max_length) then
      write(errmsg, '(*(G0))')                                                   &
        'Fatal error in Memory Manager, length of ', description, ' must be ',   &
        max_length, ' characters or less: ', name, '(len=', len(trim(name)), ')'

      ! -- store error and stop program execution
      call store_error(errmsg)
      call ustop()
    end if

  end subroutine mem_check_length
end module MemoryHelperModule