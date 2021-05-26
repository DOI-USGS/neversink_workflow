module MemoryTypeModule
  
  use KindModule, only: DP, LGP, I4B
  use ConstantsModule, only: LENMEMPATH, LENMEMADDRESS, LENTIMESERIESNAME,       &
                             LENVARNAME, MAXMEMRANK, LENMEMTYPE,                 &
                             TABSTRING, TABINTEGER,                              &
                             TABCENTER, TABLEFT, TABRIGHT
  use TableModule, only: TableType
  use MemoryHelperModule, only: create_mem_address
  
  implicit none
  private
  public :: MemoryType
 
  type MemoryType
    character(len=LENVARNAME)                              :: name                   !< name of the array
    character(len=LENVARNAME)                              :: mastername = 'none'    !< name of the master array
    character(len=LENMEMPATH)                              :: path                   !< path to memory object
    character(len=LENMEMPATH)                              :: masterPath = 'none'    !< path to master memory object
    character(len=LENMEMTYPE)                              :: memtype                !< type (INTEGER or DOUBLE)
    integer(I4B)                                           :: id                     !< id, not used
    integer(I4B)                                           :: nrealloc = 0           !< number of times reallocated
    integer(I4B)                                           :: isize                  !< size of the array
    integer(I4B)                                           :: set_handler_idx = 0    !< index of side effect handler for external access
    logical(LGP)                                           :: master = .true.        !< master copy, others point to this one
    logical(LGP), pointer                                  :: logicalsclr => null()  !< pointer to the logical
    integer(I4B), pointer                                  :: intsclr     => null()  !< pointer to the integer
    real(DP), pointer                                      :: dblsclr     => null()  !< pointer to the double
    integer(I4B), dimension(:), pointer, contiguous        :: aint1d      => null()  !< pointer to 1d integer array
    integer(I4B), dimension(:, :), pointer, contiguous     :: aint2d      => null()  !< pointer to 2d integer array
    integer(I4B), dimension(:, :, :), pointer, contiguous  :: aint3d      => null()  !< pointer to 3d integer array
    real(DP), dimension(:), pointer, contiguous            :: adbl1d      => null()  !< pointer to 1d double array
    real(DP), dimension(:, :), pointer, contiguous         :: adbl2d      => null()  !< pointer to 2d double array
    real(DP), dimension(:, :, :), pointer, contiguous      :: adbl3d      => null()  !< pointer to 3d double array
  contains
    procedure :: table_entry
    procedure :: mt_associated
  end type
  
  contains
  
  subroutine table_entry(this, memtab)
    ! -- dummy
    class(MemoryType) :: this
    type(TableType), intent(inout) :: memtab
    ! -- local
    character(len=16) :: cmem
    character(len=LENMEMADDRESS) :: cptr
    integer(I4B) :: ipos
    ! -- formats
    !
    ! -- determine memory type
    ipos = index(this%memtype, ' (')
    if (ipos < 1) then
      ipos = 16
    else
      ipos = min(16,ipos-1)
    end if
    cmem = this%memtype(1:ipos)
    !
    ! -- Set pointer string
    cptr = '--'
    if (.not. this%master) then
      cptr = create_mem_address(this%masterPath, this%mastername)
    end if
    !
    ! -- write data to the table
    call memtab%add_term(this%path)
    call memtab%add_term(this%name)
    call memtab%add_term(cmem)
    call memtab%add_term(this%isize)
    call memtab%add_term(cptr)
    !
    ! -- return
    return
  end subroutine table_entry

  function mt_associated(this) result(al)
    class(MemoryType) :: this
    logical :: al
    al = .false.
    if(associated(this%logicalsclr)) al = .true.
    if(associated(this%intsclr)) al = .true.
    if(associated(this%dblsclr)) al = .true.
    if(associated(this%aint1d)) al = .true.
    if(associated(this%aint2d)) al = .true.
    if(associated(this%aint3d)) al = .true.
    if(associated(this%adbl1d)) al = .true.
    if(associated(this%adbl2d)) al = .true. 
    if(associated(this%adbl3d)) al = .true. 
  end function mt_associated
  
end module MemoryTypeModule