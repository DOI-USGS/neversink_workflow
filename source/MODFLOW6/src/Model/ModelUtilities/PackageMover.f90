module PackageMoverModule
  
  use KindModule,          only: DP, I4B
  use ConstantsModule,     only: LENMEMPATH, DZERO
  use MemoryManagerModule, only: mem_allocate, mem_reallocate, mem_setptr,     &
                                 mem_deallocate

  implicit none
  private
  public :: PackageMoverType
  public :: set_packagemover_pointer
  public :: nulllify_packagemover_pointer
  
  type PackageMoverType
    
    character(len=LENMEMPATH)                    :: memoryPath                !< the location in the memory manager where the variables are stored
    integer(I4B), pointer                        :: nproviders
    integer(I4B), pointer                        :: nreceivers
    integer(I4B), dimension(:), pointer, contiguous :: iprmap => null()       !< map between id1 and feature (needed for lake to map from outlet to lake number)
    real(DP), dimension(:), pointer, contiguous  :: qtformvr      => null()
    real(DP), dimension(:), pointer, contiguous  :: qformvr       => null()
    real(DP), dimension(:), pointer, contiguous  :: qtomvr        => null()
    real(DP), dimension(:), pointer, contiguous  :: qfrommvr      => null()

  contains
    procedure :: ar
    procedure :: ad
    procedure :: cf
    procedure :: fc
    procedure :: da
    procedure :: allocate_scalars
    procedure :: allocate_arrays
    procedure :: get_qfrommvr
    procedure :: get_qtomvr
    procedure :: accumulate_qformvr
    
  end type PackageMoverType
  
  contains
  
  subroutine set_packagemover_pointer(packagemover, memPath)
    type(PackageMoverType), intent(inout) :: packagemover
    character(len=*), intent(in) :: memPath
    packagemover%memoryPath = memPath
    call mem_setptr(packagemover%nproviders, 'NPROVIDERS', packagemover%memoryPath)
    call mem_setptr(packagemover%nreceivers, 'NRECEIVERS', packagemover%memoryPath)
    call mem_setptr(packagemover%iprmap, 'IPRMAP', packagemover%memoryPath)
    call mem_setptr(packagemover%qtformvr, 'QTFORMVR', packagemover%memoryPath)
    call mem_setptr(packagemover%qformvr, 'QFORMVR', packagemover%memoryPath)
    call mem_setptr(packagemover%qtomvr, 'QTOMVR', packagemover%memoryPath)
    call mem_setptr(packagemover%qfrommvr, 'QFROMMVR', packagemover%memoryPath)
  end subroutine set_packagemover_pointer

  subroutine nulllify_packagemover_pointer(packagemover)
    type(PackageMoverType), intent(inout) :: packagemover
    packagemover%memoryPath = ''
    packagemover%nproviders => null()
    packagemover%nreceivers => null()
    packagemover%iprmap => null()
    packagemover%qtformvr => null()
    packagemover%qformvr => null()
    packagemover%qtomvr => null()
    packagemover%qfrommvr => null()
  end subroutine nulllify_packagemover_pointer

  subroutine ar(this, nproviders, nreceivers, memoryPath)
    class(PackageMoverType) :: this
    integer, intent(in) :: nproviders
    integer, intent(in) :: nreceivers
    character(len=*), intent(in) :: memoryPath
    !
    this%memoryPath = memoryPath
    call this%allocate_scalars()
    this%nproviders = nproviders
    this%nreceivers = nreceivers
    !
    call this%allocate_arrays()
    !
    ! -- return
    return
  end subroutine ar  
  
  subroutine ad(this)
    class(PackageMoverType) :: this
    integer :: i
    !
    ! -- set qtomvr and qformvr to zero
    do i = 1, this%nproviders
      this%qtomvr(i) = DZERO
      this%qformvr(i) = DZERO
    enddo
    !
    ! -- return
    return
  end subroutine ad
  
  subroutine cf(this)
    class(PackageMoverType) :: this
    integer :: i
    !
    ! -- set frommvr and qtomvr to zero
    do i = 1, this%nreceivers
      this%qfrommvr(i) = DZERO
    enddo
    do i = 1, this%nproviders
      this%qtomvr(i) = DZERO
      this%qtformvr(i) = this%qformvr(i)
    enddo
    !
    ! -- return
    return
  end subroutine cf
  
  subroutine fc(this)
    class(PackageMoverType) :: this
    integer :: i
    !
    ! -- set formvr to zero
    do i = 1, this%nproviders
      this%qformvr(i) = DZERO
    enddo
    !
    ! -- return
    return
  end subroutine fc
  
  subroutine da(this)
    class(PackageMoverType) :: this
    !
    ! -- arrays
    call mem_deallocate(this%iprmap)
    call mem_deallocate(this%qtformvr)
    call mem_deallocate(this%qformvr)
    call mem_deallocate(this%qtomvr)
    call mem_deallocate(this%qfrommvr)
    !
    ! -- scalars
    call mem_deallocate(this%nproviders)
    call mem_deallocate(this%nreceivers)
    !
    ! -- pointers
    nullify(this%iprmap)
    !
    ! -- return
    return
  end subroutine da
  
  subroutine allocate_scalars(this)
    class(PackageMoverType) :: this
    !
    call mem_allocate(this%nproviders, 'NPROVIDERS', this%memoryPath)
    call mem_allocate(this%nreceivers, 'NRECEIVERS', this%memoryPath)
    !
    this%nproviders = 0
    this%nreceivers = 0
    !
    ! -- return
    return
  end subroutine allocate_scalars  
  
  subroutine allocate_arrays(this)
    class(PackageMoverType) :: this
    integer(I4B) :: i
    !
    call mem_allocate(this%iprmap, this%nproviders, 'IPRMAP', this%memoryPath)
    call mem_allocate(this%qtformvr, this%nproviders, 'QTFORMVR', this%memoryPath)
    call mem_allocate(this%qformvr, this%nproviders, 'QFORMVR', this%memoryPath)
    call mem_allocate(this%qtomvr, this%nproviders, 'QTOMVR', this%memoryPath)
    call mem_allocate(this%qfrommvr, this%nreceivers, 'QFROMMVR', this%memoryPath)
    !
    ! -- initialize
    do i = 1, this%nproviders
      this%iprmap(i) = i
      this%qtformvr(i) = DZERO
      this%qformvr(i) = DZERO
      this%qtomvr(i) = DZERO
    enddo
    do i = 1, this%nreceivers
      this%qfrommvr(i) = DZERO
    enddo
    !
    ! -- return
    return
  end subroutine allocate_arrays
  
  function get_qfrommvr(this, ireceiver) result(qfrommvr)
    class(PackageMoverType) :: this
    real(DP) :: qfrommvr
    integer, intent(in) :: ireceiver
    qfrommvr = this%qfrommvr(ireceiver)
  end function get_qfrommvr
  
  function get_qtomvr(this, iprovider) result(qtomvr)
    class(PackageMoverType) :: this
    real(DP) :: qtomvr
    integer, intent(in) :: iprovider
    qtomvr = this%qtomvr(iprovider)
  end function get_qtomvr

  subroutine accumulate_qformvr(this, iprovider, qformvr)
    class(PackageMoverType) :: this
    integer, intent(in) :: iprovider
    real(DP), intent(in) :: qformvr
    this%qformvr(iprovider) = this%qformvr(iprovider) + qformvr
  end subroutine accumulate_qformvr

end module PackageMoverModule