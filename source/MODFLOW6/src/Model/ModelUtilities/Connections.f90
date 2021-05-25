module ConnectionsModule

  use ArrayReadersModule, only: ReadArray
  use KindModule, only: DP, I4B
  use ConstantsModule, only: LENMODELNAME, LENMEMPATH
  use GenericUtilitiesModule, only: sim_message
  use BlockParserModule, only: BlockParserType
  
  implicit none
  private
  public :: ConnectionsType
  public :: iac_to_ia

  type ConnectionsType
    character(len=LENMEMPATH)                       :: memoryPath                !< memory path of the connections data
    character(len=LENMODELNAME), pointer            :: name_model => null()      !< name of the model    
    integer(I4B), pointer                           :: nodes      => null()      !< number of nodes
    integer(I4B), pointer                           :: nja        => null()      !< number of connections
    integer(I4B), pointer                           :: njas       => null()      !< number of symmetric connections
    integer(I4B), pointer                           :: ianglex    => null()      !< indicates whether or not anglex was read
    integer(I4B), dimension(:), pointer, contiguous :: ia         => null()      !< (size:nodes+1) csr index array
    integer(I4B), dimension(:), pointer, contiguous :: ja         => null()      !< (size:nja) csr pointer array
    integer(I4B), dimension(:), pointer, contiguous :: mask       => null()      !< (size:nja) to mask certain connections: ==0 means masked. Do not set the mask directly, use set_mask instead!    
    real(DP), dimension(:), pointer, contiguous     :: cl1        => null()      !< (size:njas) connection length between node n and shared face with node m
    real(DP), dimension(:), pointer, contiguous     :: cl2        => null()      !< (size:njas) connection length between node m and shared face with node n
    real(DP), dimension(:), pointer, contiguous     :: hwva       => null()      !< (size:njas) horizontal perpendicular width (ihc>0) or vertical flow area (ihc=0)
    real(DP), dimension(:), pointer, contiguous     :: anglex     => null()      !< (size:njas) connection angle of face normal with x axis (read in degrees, stored as radians)
    integer(I4B), dimension(:), pointer, contiguous :: isym       => null()      !< (size:nja) returns csr index of symmetric counterpart
    integer(I4B), dimension(:), pointer, contiguous :: jas        => null()      !< (size:nja) map any connection to upper triangle (for pulling out of symmetric array)
    integer(I4B), dimension(:), pointer, contiguous :: ihc        => null()      !< (size:njas) horizontal connection (0:vertical, 1:mean thickness, 2:staggered)
    integer(I4B), dimension(:), pointer, contiguous :: iausr      => null()      !< (size:nodesusr+1) 
    integer(I4B), dimension(:), pointer, contiguous :: jausr      => null()      !< (size:nja)
    type(BlockParserType)                           :: parser                    !< block parser
  contains
    procedure :: con_da
    procedure :: allocate_scalars
    procedure :: allocate_arrays
    procedure :: con_finalize
    procedure :: read_connectivity_from_block
    procedure :: set_cl1_cl2_from_fleng
    procedure :: disconnections
    procedure :: disvconnections
    procedure :: disuconnections
    procedure :: iajausr
    procedure :: getjaindex
    procedure :: set_mask
  end type ConnectionsType

  contains
  
  subroutine con_da(this)
! ******************************************************************************
! con_da -- Deallocate connection variables
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_deallocate
    ! -- dummy
    class(ConnectionsType) :: this
! ------------------------------------------------------------------------------
    !
    ! -- Strings
    deallocate(this%name_model)
    !
    ! -- Scalars
    call mem_deallocate(this%nodes)
    call mem_deallocate(this%nja)
    call mem_deallocate(this%njas)
    call mem_deallocate(this%ianglex)
    !
    ! -- iausr and jausr
    if(associated(this%iausr, this%ia)) then
      nullify(this%iausr)
    else
      call mem_deallocate(this%iausr)
    endif
    if(associated(this%jausr, this%ja)) then
      nullify(this%jausr)
    else
      call mem_deallocate(this%jausr)
    endif
    ! -- mask
    if (associated(this%mask, this%ja)) then
      nullify(this%mask)
    else
      call mem_deallocate(this%mask)
    end if 
    !
    ! -- Arrays
    call mem_deallocate(this%ia)
    call mem_deallocate(this%ja)  
    call mem_deallocate(this%isym)
    call mem_deallocate(this%jas)
    call mem_deallocate(this%hwva)
    call mem_deallocate(this%anglex)
    call mem_deallocate(this%ihc)
    call mem_deallocate(this%cl1)
    call mem_deallocate(this%cl2)     
    !
    ! -- return
    return
  end subroutine con_da
  
  subroutine allocate_scalars(this, name_model)
! ******************************************************************************
! allocate_scalars -- Allocate scalars for ConnectionsType
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_allocate
    use MemoryHelperModule, only: create_mem_path
    ! -- dummy
    class(ConnectionsType) :: this
    character(len=*), intent(in) :: name_model
! ------------------------------------------------------------------------------
    !
    ! -- allocate
    allocate(this%name_model)
    
    this%memoryPath = create_mem_path(name_model, 'CON')
    call mem_allocate(this%nodes, 'NODES', this%memoryPath)
    call mem_allocate(this%nja, 'NJA', this%memoryPath)
    call mem_allocate(this%njas, 'NJAS', this%memoryPath)
    call mem_allocate(this%ianglex, 'IANGLEX', this%memoryPath)
    this%name_model = name_model
    this%nodes = 0
    this%nja = 0
    this%njas = 0
    this%ianglex = 0
    !
    ! -- Return
    return
  end subroutine allocate_scalars

  subroutine allocate_arrays(this)
! ******************************************************************************
! allocate_arrays -- Allocate arrays for ConnectionsType
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use MemoryManagerModule, only: mem_allocate
    ! -- dummy
    class(ConnectionsType) :: this
! ------------------------------------------------------------------------------
    !
    ! -- allocate space for connection arrays
    call mem_allocate(this%ia, this%nodes+1, 'IA', this%memoryPath)
    call mem_allocate(this%ja, this%nja, 'JA', this%memoryPath)
    call mem_allocate(this%isym, this%nja, 'ISYM', this%memoryPath)
    call mem_allocate(this%jas, this%nja, 'JAS', this%memoryPath)
    call mem_allocate(this%hwva, this%njas, 'HWVA', this%memoryPath)
    call mem_allocate(this%anglex, this%njas, 'ANGLEX', this%memoryPath)
    call mem_allocate(this%ihc, this%njas, 'IHC', this%memoryPath)
    call mem_allocate(this%cl1, this%njas, 'CL1', this%memoryPath)
    call mem_allocate(this%cl2, this%njas, 'CL2', this%memoryPath)
    call mem_allocate(this%iausr, 1, 'IAUSR', this%memoryPath)
    call mem_allocate(this%jausr, 1, 'JAUSR', this%memoryPath)
    ! 
    ! -- let mask point to ja, which is always nonzero, 
    !    until someone decides to do a 'set_mask'
    this%mask => this%ja
    !
    ! -- Return
    return
  end subroutine allocate_arrays
  
  subroutine con_finalize(this, ihctemp, cl12temp, hwvatemp, angldegx)
! ******************************************************************************
! con_finalize -- Finalize connection data
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH, DONE, DHALF, DPIO180, DNODATA
    use SimModule, only: ustop, store_error, count_errors, store_error_unit
    ! -- dummy
    class(ConnectionsType) :: this
    integer(I4B), dimension(:), intent(in) :: ihctemp
    real(DP), dimension(:), intent(in) :: cl12temp
    real(DP), dimension(:), intent(in) :: hwvatemp
    real(DP), dimension(:), intent(in) :: angldegx
    ! -- local
    character(len=LINELENGTH) :: errmsg
    integer(I4B) :: ii, n, m
    integer(I4B), parameter :: nname = 6
    character(len=24),dimension(nname) :: aname(nname)
    ! -- formats
    character(len=*),parameter :: fmtsymerr =                                  &
      &"('Error in array: ',a,'.',                                             &
        &' Array is not symmetric in positions: ',i0,' and ',i0,'.',           &
        &' Values in these positions are: ',1pg15.6,' and ', 1pg15.6,          &
        &' For node ',i0,' connected to node ',i0)"
    character(len=*),parameter :: fmtsymerrja =                                &
      &"('Error in array: ',a,'.',                                             &
        &' Array does not have symmetric counterpart in position ',i0,         &
        &' for cell ',i0,' connected to cell ',i0)"
    character(len=*),parameter :: fmtjanmerr =                                 &
      &"('Error in array: ',a,'.',                                             &
        &' First value for cell : ',i0,' must equal ',i0,'.',                  &
        &' Found ',i0,' instead.')"
    character(len=*),parameter :: fmtjasorterr =                               &
      &"('Error in array: ',a,'.',                                             &
        &' Entries not sorted for row: ',i0,'.',                               &
        &' Offending entries are: ',i0,' and ',i0)"
    character(len=*),parameter :: fmtihcerr =                                  &
      "('IHC must be 0, 1, or 2.  Found: ',i0)"
    ! -- data
    data aname(1) /'                     IAC'/
    data aname(2) /'                      JA'/
    data aname(3) /'                     IHC'/
    data aname(4) /'                    CL12'/
    data aname(5) /'                    HWVA'/
    data aname(6) /'                ANGLDEGX'/
! ------------------------------------------------------------------------------
    !
    ! -- Convert any negative ja numbers to positive
    do ii = 1, this%nja
      if(this%ja(ii) < 0) this%ja(ii) = -this%ja(ii)
    enddo
    !
    ! -- Ensure ja is sorted with the row column listed first
    do n = 1, this%nodes
      m = this%ja(this%ia(n))
      if (n /= m) then
        write(errmsg, fmtjanmerr) trim(adjustl(aname(2))), n, n, m
        call store_error(errmsg)
      endif
      do ii = this%ia(n) + 1, this%ia(n + 1) - 2
        m = this%ja(ii)
        if(m > this%ja(ii+1)) then
          write(errmsg, fmtjasorterr) trim(adjustl(aname(2))), n,              &
                                      m, this%ja(ii+1)
          call store_error(errmsg)
        endif
      enddo
    enddo
    if(count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- fill the isym arrays
    call fillisym(this%nodes, this%nja, this%ia, this%ja, this%isym)
    !
    ! -- check for symmetry in ja (isym value of zero indicates there is no
    !    symmetric connection
    do n = 1, this%nodes
      do ii = this%ia(n), this%ia(n + 1) - 1
        m = this%ja(ii)
        if(this%isym(ii) == 0) then
          write(errmsg, fmtsymerrja) trim(adjustl(aname(2))), ii, n, m
          call store_error(errmsg)
        endif
      enddo
    enddo
    if(count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Fill the jas array, which maps any connection to upper triangle
    call filljas(this%nodes, this%nja, this%ia, this%ja, this%isym, this%jas)
    !
    ! -- Put into symmetric array
    do n = 1, this%nodes
      do ii = this%ia(n) + 1, this%ia(n + 1) - 1
        m = this%ja(ii)
        if(ihctemp(ii) /= ihctemp(this%isym(ii))) then
          write(errmsg, fmtsymerr) trim(adjustl(aname(3))), ii, this%isym(ii), &
                             ihctemp(ii), ihctemp(this%isym(ii)), n, m
          call store_error(errmsg)
        else
          this%ihc(this%jas(ii)) = ihctemp(ii)
        endif
      enddo
    enddo
    if(count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Put cl12 into symmetric arrays cl1 and cl2
    do n = 1, this%nodes
      do ii = this%ia(n) + 1, this%ia(n + 1) - 1
        m = this%ja(ii)
        if(m > n) then
          this%cl1(this%jas(ii)) = cl12temp(ii)
        elseif(n > m) then
          this%cl2(this%jas(ii)) = cl12temp(ii)
        endif
      enddo
    enddo
    !
    ! -- Put HWVA into symmetric array based on the value of IHC
    !    IHC = 0, vertical connection, HWVA is vertical flow area
    !    IHC = 1, horizontal connection, HWVA is the width perpendicular to
    !             flow
    !    IHC = 2, horizontal connection for a vertically staggered grid.
    !             HWVA is the width perpendicular to flow.
    do n = 1, this%nodes
      do ii = this%ia(n) + 1, this%ia(n + 1) - 1
        m = this%ja(ii)
        if(hwvatemp(ii) /= hwvatemp(this%isym(ii))) then
          write(errmsg, fmtsymerr) trim(adjustl(aname(5))), ii, this%isym(ii), &
                             hwvatemp(ii), hwvatemp(this%isym(ii)), n, m
          call store_error(errmsg)
        endif
        if(ihctemp(ii) < 0 .or. ihctemp(ii) > 2) then
          write(errmsg, fmtihcerr) ihctemp(ii)
          call store_error(errmsg)
        endif
        this%hwva(this%jas(ii)) = hwvatemp(ii)
      enddo
    enddo
    if(count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Put anglextemp into this%anglex; store only upper triangle
    if(this%ianglex /= 0) then
      do n = 1, this%nodes
        do ii = this%ia(n) + 1, this%ia(n + 1) - 1
          m = this%ja(ii)
          if(n > m) cycle
          this%anglex(this%jas(ii)) = angldegx(ii) * DPIO180
        enddo
      enddo
    else
      do n = 1, size(this%anglex)
        this%anglex(n) = DNODATA
      enddo
    endif
    !
    ! -- Return
    return
  end subroutine con_finalize
    
  subroutine read_connectivity_from_block(this, name_model, nodes, nja, iout)
! ******************************************************************************
! read_connectivity_from_block -- Read and process IAC and JA from an
! an input block called CONNECTIONDATA
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use SimModule, only: ustop, store_error, count_errors
    ! -- dummy
    class(ConnectionsType) :: this
    character(len=*), intent(in) :: name_model
    integer(I4B), intent(in) :: nodes
    integer(I4B), intent(in) :: nja
    integer(I4B), intent(in) :: iout
    ! -- local
    character(len=LINELENGTH) :: line
    character(len=LINELENGTH) :: keyword
    integer(I4B) :: ii,n,m
    integer(I4B) :: ierr, nerr
    logical :: isfound, endOfBlock
    integer(I4B), parameter :: nname = 2
    logical,dimension(nname) :: lname
    character(len=24),dimension(nname) :: aname(nname)
    character(len=300) :: ermsg
    ! -- formats
    character(len=*),parameter :: fmtsymerr = &
      &"(/,'Error in array: ',(a),/, &
          &'Array is not symmetric in positions: ',2i9,/, &
          &'Values in these positions are: ', 2(1pg15.6))"
    character(len=*),parameter :: fmtihcerr = &
      &"(/,'IHC must be 0, 1, or 2.  Found: ',i0)"
    ! -- data
    data aname(1) /'                     IAC'/
    data aname(2) /'                      JA'/
! ------------------------------------------------------------------------------
    !
    ! -- Allocate and initialize dimensions
    call this%allocate_scalars(name_model)
    this%nodes = nodes
    this%nja = nja
    this%njas = (this%nja - this%nodes) / 2
    !
    ! -- Allocate space for connection arrays
    call this%allocate_arrays()
    !
    ! -- get connectiondata block
    call this%parser%GetBlock('CONNECTIONDATA', isfound, ierr)
    lname(:) = .false.
    if(isfound) then
      write(iout,'(1x,a)')'PROCESSING CONNECTIONDATA'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(keyword)
        select case (keyword)
          case ('IAC')
            call ReadArray(this%parser%iuactive, this%ia, aname(1), 1, &
                            this%nodes, iout, 0)
            lname(1) = .true.
          case ('JA')
            call ReadArray(this%parser%iuactive, this%ja, aname(2), 1, &
                            this%nja, iout, 0)
            lname(2) = .true.
          case default
            write(ermsg,'(4x,a,a)')'ERROR. UNKNOWN CONNECTIONDATA TAG: ', &
                                     trim(keyword)
            call store_error(ermsg)
            call this%parser%StoreErrorUnit()
            call ustop()
        end select
      end do
      write(iout,'(1x,a)')'END PROCESSING CONNECTIONDATA'
    else
      call store_error('ERROR.  REQUIRED CONNECTIONDATA BLOCK NOT FOUND.')
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- verify all items were read
    do n = 1, nname
      if(.not. lname(n)) then
        write(ermsg,'(1x,a,a)') &
          'ERROR.  REQUIRED INPUT WAS NOT SPECIFIED: ',aname(n)
        call store_error(ermsg)
      endif
    enddo
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Convert iac to ia
    do n = 2, this%nodes + 1
      this%ia(n) = this%ia(n) + this%ia(n-1)
    enddo
    do n = this%nodes + 1, 2, -1
      this%ia(n) = this%ia(n - 1) + 1
    enddo
    this%ia(1) = 1
    !
    ! -- Convert any negative ja numbers to positive
    do ii = 1, this%nja
      if(this%ja(ii) < 0) this%ja(ii) = -this%ja(ii)
    enddo
    !
    ! -- fill the isym and jas arrays
    call fillisym(this%nodes, this%nja, this%ia, this%ja, this%isym)
    call filljas(this%nodes, this%nja, this%ia, this%ja, this%isym, &
                 this%jas)
    !
    ! -- check for symmetry in ja
    do n = 1, this%nodes
      do ii = this%ia(n), this%ia(n + 1) - 1
        m = this%ja(ii)
        if(n /= this%ja(this%isym(ii))) then
          write(line, fmtsymerr) aname(2), ii, this%isym(ii)
          call sim_message(line)
          call this%parser%StoreErrorUnit()
          call ustop()
        endif
      enddo
    enddo
    !
    nerr = count_errors()
    if(nerr > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Return
    return
  end subroutine read_connectivity_from_block
  
  subroutine set_cl1_cl2_from_fleng(this, fleng)
! ******************************************************************************
! set_cl1_cl2_from_fleng -- Using a vector of cell lengths, 
! calculate the cl1 and cl2 arrays.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DHALF
    ! -- dummy
    class(ConnectionsType) :: this
    real(DP), dimension(:), intent(in) :: fleng
    ! -- local
    integer(I4B) :: n, m, ii
! ------------------------------------------------------------------------------
    !
    ! -- Fill symmetric arrays cl1 and cl2 from fleng of the node
    do n = 1, this%nodes
      do ii = this%ia(n) + 1, this%ia(n + 1) - 1
        m = this%ja(ii)
        this%cl1(this%jas(ii)) = fleng(n) * DHALF
        this%cl2(this%jas(ii)) = fleng(m) * DHALF
      enddo
    enddo
    !
    ! -- Return
    return
  end subroutine set_cl1_cl2_from_fleng
  
  subroutine disconnections(this, name_model, nodes, ncol, nrow, nlay,         &
                            nrsize, delr, delc, top, bot, nodereduced,         &
                            nodeuser)
! ******************************************************************************
! disconnections -- Construct the connectivity arrays for a structured
!   three-dimensional grid.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DHALF, DZERO, DTHREE, DTWO, DPI
    use SparseModule, only: sparsematrix
    use InputOutputModule, only: get_node
    ! -- dummy
    class(ConnectionsType) :: this
    character(len=*), intent(in) :: name_model
    integer(I4B), intent(in) :: nodes
    integer(I4B), intent(in) :: ncol
    integer(I4B), intent(in) :: nrow
    integer(I4B), intent(in) :: nlay
    integer(I4B), intent(in) :: nrsize
    real(DP), dimension(ncol),             intent(in) :: delr
    real(DP), dimension(nrow),             intent(in) :: delc
    real(DP), dimension(nodes),            intent(in) :: top
    real(DP), dimension(nodes),            intent(in) :: bot
    integer(I4B),          dimension(:), target,        intent(in) :: nodereduced
    integer(I4B),          dimension(:),                intent(in) :: nodeuser
    ! -- local
    integer(I4B), dimension(:, :, :), pointer :: nrdcd_ptr => null() !non-contiguous because is a slice
    integer(I4B), dimension(:), allocatable :: rowmaxnnz
    type(sparsematrix) :: sparse
    integer(I4B) :: i, j, k, kk, ierror, isympos, nodesuser
    integer(I4B) :: nr, mr
! ------------------------------------------------------------------------------
    !
    ! -- Allocate scalars
    call this%allocate_scalars(name_model)
    !
    ! -- Set scalars
    this%nodes = nodes
    this%ianglex = 1
    !
    ! -- Setup the sparse matrix object
    allocate(rowmaxnnz(this%nodes))
    do i = 1, this%nodes
      rowmaxnnz(i) = 6
    enddo
    call sparse%init(this%nodes, this%nodes, rowmaxnnz)
    !
    ! -- Create a 3d pointer to nodereduced for easier processing
    if(nrsize /= 0) then
      nrdcd_ptr(1:ncol, 1:nrow, 1:nlay) => nodereduced
    endif
    !
    ! -- Add connections to sparse
    do k = 1, nlay
      do i = 1, nrow
        do j = 1, ncol
          !
          ! -- Find the reduced node number and then cycle if the
          !    node is always inactive
          if(nrsize == 0) then
            nr = get_node(k, i, j, nlay, nrow, ncol)
          else
            nr = nrdcd_ptr(j, i, k)
          endif
          if(nr <= 0) cycle
          !
          ! -- Process diagonal
          call sparse%addconnection(nr, nr, 1)
          !
          ! -- Up direction
          if(k > 1) then
            do kk = k - 1, 1, -1
              if(nrsize == 0) then
                mr = get_node(kk, i, j, nlay, nrow, ncol)
              else
                mr = nrdcd_ptr(j, i, kk)
              endif
              if(mr >= 0) exit
            enddo
            if(mr > 0) then
              call sparse%addconnection(nr, mr, 1)
            endif
          endif
          !
          ! -- Back direction
          if(i > 1) then
            if(nrsize == 0) then
              mr = get_node(k, i-1, j, nlay, nrow, ncol)
            else
              mr = nrdcd_ptr(j, i-1, k)
            endif
            if(mr > 0) then
              call sparse%addconnection(nr, mr, 1)
            endif
          endif
          !
          ! -- Left direction
          if(j > 1) then
            if(nrsize == 0) then
              mr = get_node(k, i, j-1, nlay, nrow, ncol)
            else
              mr = nrdcd_ptr(j-1, i, k)
            endif
            if(mr > 0) then
              call sparse%addconnection(nr, mr, 1)
            endif
          endif
          !
          ! -- Right direction
          if(j < ncol) then
              if(nrsize == 0) then
                mr = get_node(k, i, j+1, nlay, nrow, ncol)
              else
                mr = nrdcd_ptr(j+1, i, k)
              endif
            if(mr > 0) then
              call sparse%addconnection(nr, mr, 1)
            endif
          endif
          !
          ! -- Front direction
          if(i < nrow) then  !front
              if(nrsize == 0) then
                mr = get_node(k, i+1, j, nlay, nrow, ncol)
              else
                mr = nrdcd_ptr(j, i+1, k)
              endif
            if(mr > 0) then
              call sparse%addconnection(nr, mr, 1)
            endif
          endif
          !
          ! -- Down direction
          if(k < nlay) then
            do kk = k + 1, nlay
              if(nrsize == 0) then
                mr = get_node(kk, i, j, nlay, nrow, ncol)
              else
                mr = nrdcd_ptr(j, i, kk)
              endif
              if(mr >= 0) exit
            enddo
            if(mr > 0) then
              call sparse%addconnection(nr, mr, 1)
            endif
          endif
        enddo
      enddo
    enddo
    this%nja = sparse%nnz
    this%njas = (this%nja - this%nodes) / 2
    !
    ! -- Allocate index arrays of size nja and symmetric arrays
    call this%allocate_arrays()
    !
    ! -- Fill the IA and JA arrays from sparse, then destroy sparse
    call sparse%filliaja(this%ia, this%ja, ierror)
    call sparse%destroy()
    !
    ! -- fill the isym and jas arrays
    call fillisym(this%nodes, this%nja, this%ia, this%ja, this%isym)
    call filljas(this%nodes, this%nja, this%ia, this%ja, this%isym, this%jas)
    !
    ! -- Fill symmetric discretization arrays (ihc,cl1,cl2,hwva,anglex)
    isympos = 1
    do k = 1, nlay
      do i = 1, nrow
        do j = 1, ncol
          !
          ! -- cycle if node is always inactive
          if(nrsize == 0) then
            nr = get_node(k, i, j, nlay, nrow, ncol)
          else
            nr = nrdcd_ptr(j, i, k)
          endif
          if(nr <= 0) cycle
          !
          ! -- right connection
          if(j < ncol) then
            if(nrsize == 0) then
              mr = get_node(k, i, j+1, nlay, nrow, ncol)
            else
              mr = nrdcd_ptr(j+1, i, k)
            endif
            if(mr > 0) then
              this%ihc(isympos) = 1
              this%cl1(isympos) = DHALF * delr(j)
              this%cl2(isympos) = DHALF * delr(j + 1)
              this%hwva(isympos) = delc(i)
              this%anglex(isympos) = DZERO
              isympos = isympos + 1
            endif
          endif
          !
          ! -- front connection
          if(i < nrow) then
            if(nrsize == 0) then
              mr = get_node(k, i+1, j, nlay, nrow, ncol)
            else
              mr = nrdcd_ptr(j, i+1, k)
            endif
            if(mr > 0) then
              this%ihc(isympos) = 1
              this%cl1(isympos) = DHALF * delc(i)
              this%cl2(isympos) = DHALF * delc(i + 1)
              this%hwva(isympos) = delr(j)
              this%anglex(isympos) = DTHREE / DTWO * DPI
              isympos = isympos + 1
            endif
          endif
          !
          ! -- down connection
          if(k < nlay) then
            do kk = k + 1, nlay
              if(nrsize == 0) then
                mr = get_node(kk, i, j, nlay, nrow, ncol)
              else
                mr = nrdcd_ptr(j, i, kk)
              endif
              if(mr >= 0) exit
            enddo
            if(mr > 0) then
              this%ihc(isympos) = 0
              this%cl1(isympos) = DHALF * (top(nr) - bot(nr))
              this%cl2(isympos) = DHALF * (top(mr) - bot(mr))
              this%hwva(isympos) = delr(j) * delc(i)
              this%anglex(isympos) = DZERO
              isympos = isympos + 1
            endif
          endif
        enddo
      enddo
    enddo
    !
    ! -- Deallocate temporary arrays
    deallocate(rowmaxnnz)
    !
    ! -- If reduced system, then need to build iausr and jausr, otherwise point
    !    them to ia and ja.
    nodesuser = nlay * nrow * ncol
    call this%iajausr(nrsize, nodesuser, nodereduced, nodeuser)
    !
    ! -- Return
    return
  end subroutine disconnections

  subroutine disvconnections(this, name_model, nodes, ncpl, nlay, nrsize,      &
                             nvert, vertex, iavert, javert, cellxy,            & 
                             top, bot, nodereduced, nodeuser)
! ******************************************************************************
! disvconnections -- Construct the connectivity arrays using cell disv
!   information.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DHALF, DZERO, DTHREE, DTWO, DPI
    use SparseModule, only: sparsematrix
    use InputOutputModule, only: get_node
    use DisvGeom, only: DisvGeomType
    use MemoryManagerModule, only: mem_reallocate
    ! -- dummy
    class(ConnectionsType)                              :: this
    character(len=*),                        intent(in) :: name_model
    integer(I4B),                                 intent(in) :: nodes
    integer(I4B),                                 intent(in) :: ncpl
    integer(I4B),                                 intent(in) :: nlay
    integer(I4B),                                 intent(in) :: nrsize
    integer(I4B),                                 intent(in) :: nvert
    real(DP), dimension(2, nvert),   intent(in) :: vertex
    integer(I4B), dimension(:),                   intent(in) :: iavert
    integer(I4B), dimension(:),                   intent(in) :: javert
    real(DP), dimension(2, ncpl),    intent(in) :: cellxy
    real(DP), dimension(nodes),      intent(in) :: top
    real(DP), dimension(nodes),      intent(in) :: bot
    integer(I4B),          dimension(:),          intent(in) :: nodereduced
    integer(I4B),          dimension(:),          intent(in) :: nodeuser
    ! -- local
    integer(I4B), dimension(:), allocatable :: itemp
    type(sparsematrix) :: sparse, vertcellspm
    integer(I4B) :: n, m, ipos, i, j, ierror, nodesuser
    type(DisvGeomType) :: cell1, cell2
! ------------------------------------------------------------------------------
    !
    ! -- Allocate scalars
    call this%allocate_scalars(name_model)
    !
    ! -- Set scalars
    this%nodes = nodes
    this%ianglex = 1
    !
    ! -- Initialize DisvGeomType objects
    call cell1%init(nlay, ncpl, nodes, top, bot, iavert, javert, vertex,       &
                    cellxy, nodereduced, nodeuser)
    call cell2%init(nlay, ncpl, nodes, top, bot, iavert, javert, vertex,       &
                    cellxy, nodereduced, nodeuser)
    !
    ! -- Create a sparse matrix array with a row for each vertex.  The columns
    !    in the sparse matrix contains the cells that include that vertex.
    !    This array will be used to determine horizontal cell connectivity.
    allocate(itemp(nvert))
    do i = 1, nvert
      itemp(i) = 4
    enddo
    call vertcellspm%init(nvert, ncpl, itemp)
    deallocate(itemp)
    do j = 1, ncpl
      do i = iavert(j), iavert(j + 1) - 1
        call vertcellspm%addconnection(javert(i), j, 1)
      enddo
    enddo
    !
    ! -- Call routine to build a sparse matrix of the connections
    call vertexconnect(this%nodes, nrsize, 6, nlay, ncpl, sparse,              &
                       vertcellspm, cell1, cell2, nodereduced)
    this%nja = sparse%nnz
    this%njas = (this%nja - this%nodes) / 2
    !
    ! -- Allocate index arrays of size nja and symmetric arrays
    call this%allocate_arrays()
    !
    ! -- Fill the IA and JA arrays from sparse, then destroy sparse
    call sparse%sort()
    call sparse%filliaja(this%ia, this%ja, ierror)
    call sparse%destroy()
    !
    ! -- fill the isym and jas arrays
    call fillisym(this%nodes, this%nja, this%ia, this%ja, this%isym)
    call filljas(this%nodes, this%nja, this%ia, this%ja, this%isym, this%jas)
    !
    ! -- Fill symmetric discretization arrays (ihc,cl1,cl2,hwva,anglex)
    do n = 1, this%nodes
      call cell1%set_nodered(n)
      do ipos = this%ia(n) + 1, this%ia(n + 1) - 1
        m = this%ja(ipos)
        if(m < n) cycle
        call cell2%set_nodered(m)
        call cell1%cprops(cell2, this%hwva(this%jas(ipos)),                    &
                          this%cl1(this%jas(ipos)), this%cl2(this%jas(ipos)),  &
                          this%anglex(this%jas(ipos)),                         &
                          this%ihc(this%jas(ipos)))
      enddo
    enddo
    !
    ! -- If reduced system, then need to build iausr and jausr, otherwise point
    !    them to ia and ja.
    nodesuser = nlay * ncpl
    call this%iajausr(nrsize, nodesuser, nodereduced, nodeuser)
    !
    ! -- Return
    return
  end subroutine disvconnections

  subroutine disuconnections(this, name_model, nodes, nodesuser, nrsize, &
                             nodereduced, nodeuser, iainp, jainp, &
                             ihcinp, cl12inp, hwvainp, angldegxinp)
! ******************************************************************************
! disuconnections -- Construct the connectivity arrays using disu
!   information.  Grid may be reduced
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DHALF, DZERO, DTHREE, DTWO, DPI
    use SparseModule, only: sparsematrix
    use MemoryManagerModule, only: mem_reallocate
    ! -- dummy
    class(ConnectionsType) :: this
    character(len=*), intent(in) :: name_model
    integer(I4B), intent(in) :: nodes
    integer(I4B), intent(in) :: nodesuser
    integer(I4B), intent(in) :: nrsize
    integer(I4B), dimension(:), contiguous, intent(in) :: nodereduced
    integer(I4B), dimension(:), contiguous, intent(in) :: nodeuser
    integer(I4B), dimension(:), contiguous, intent(in) :: iainp
    integer(I4B), dimension(:), contiguous, intent(in) :: jainp
    integer(I4B), dimension(:), contiguous, intent(in) :: ihcinp
    real(DP), dimension(:), contiguous, intent(in) :: cl12inp
    real(DP), dimension(:), contiguous, intent(in) :: hwvainp
    real(DP), dimension(:), contiguous, intent(in) :: angldegxinp
    ! -- local
    integer(I4B),dimension(:),allocatable :: ihctemp
    real(DP),dimension(:),allocatable :: cl12temp
    real(DP),dimension(:),allocatable :: hwvatemp
    real(DP),dimension(:),allocatable :: angldegxtemp
    integer(I4B) :: nr, nu, mr, mu, ipos, iposr, ierror
    integer(I4B), dimension(:), allocatable :: rowmaxnnz
    type(sparsematrix) :: sparse
! ------------------------------------------------------------------------------
    !
    ! -- Allocate scalars
    call this%allocate_scalars(name_model)
    !
    ! -- Set scalars
    this%nodes = nodes
    this%ianglex = 1
    !
    ! -- If not a reduced grid, then copy and finalize, otherwise more
    !    processing is required
    if (nrsize == 0) then
      this%nodes = nodes
      this%nja = size(jainp)
      this%njas = (this%nja - this%nodes) / 2
      call this%allocate_arrays()
      do nu = 1, nodes + 1
        this%ia(nu) = iainp(nu)
      end do
      do ipos = 1, this%nja
        this%ja(ipos) = jainp(ipos)
      end do
      !
      ! -- Call con_finalize to check inp arrays and push larger arrays
      !    into compressed symmetric arrays
      call this%con_finalize(ihcinp, cl12inp, hwvainp, angldegxinp)
      !
    else
      ! -- reduced system requires more work
      !
      ! -- Setup the sparse matrix object
      allocate(rowmaxnnz(this%nodes))
      do nr = 1, this%nodes
        nu = nodeuser(nr)
        rowmaxnnz(nr) = iainp(nu + 1) - iainp(nu)
      enddo
      call sparse%init(this%nodes, this%nodes, rowmaxnnz)
      !
      ! -- go through user connectivity and create sparse
      do nu = 1, nodesuser
        nr = nodereduced(nu)
        if (nr > 0) call sparse%addconnection(nr, nr, 1)
        do ipos = iainp(nu) + 1, iainp(nu + 1) - 1
          mu = jainp(ipos)
          mr = nodereduced(mu)
          if (nr < 1) cycle
          if (mr < 1) cycle
          call sparse%addconnection(nr, mr, 1)
        enddo
      enddo
      this%nja = sparse%nnz
      this%njas = (this%nja - this%nodes) / 2
      !
      ! -- Allocate index arrays of size nja and symmetric arrays
      call this%allocate_arrays()
      !
      ! -- Fill the IA and JA arrays from sparse, then destroy sparse
      call sparse%sort()
      call sparse%filliaja(this%ia, this%ja, ierror)
      call sparse%destroy()
      deallocate(rowmaxnnz)
      !
      ! -- At this point, need to reduce ihc, cl12, hwva, and angldegx
      allocate(ihctemp(this%nja))
      allocate(cl12temp(this%nja))
      allocate(hwvatemp(this%nja))
      allocate(angldegxtemp(this%nja))
      !
      ! -- Compress user arrays into reduced arrays
      iposr = 1
      do nu = 1, nodesuser
        nr = nodereduced(nu)
        do ipos = iainp(nu), iainp(nu + 1) - 1
          mu = jainp(ipos)
          mr = nodereduced(mu)
          if (nr < 1 .or. mr < 1) cycle
          ihctemp(iposr) = ihcinp(ipos)
          cl12temp(iposr) = cl12inp(ipos)
          hwvatemp(iposr) = hwvainp(ipos)
          angldegxtemp(iposr) = angldegxinp(ipos)
          iposr = iposr + 1
        end do
      end do
      !
      ! -- call finalize
      call this%con_finalize(ihctemp, cl12temp, hwvatemp, angldegxtemp)
      !
      ! -- deallocate temporary arrays
      deallocate(ihctemp)
      deallocate(cl12temp)
      deallocate(hwvatemp)
      deallocate(angldegxtemp)
    end if
    !
    ! -- If reduced system, then need to build iausr and jausr, otherwise point
    !    them to ia and ja.
    call this%iajausr(nrsize, nodesuser, nodereduced, nodeuser)
    !
    ! -- Return
    return
  end subroutine disuconnections

  subroutine iajausr(this, nrsize, nodesuser, nodereduced, nodeuser)
! ******************************************************************************
! iajausr -- Fill iausr and jausr if reduced grid, otherwise point them
!   to ia and ja.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_reallocate, mem_deallocate, mem_setptr
    ! -- dummy
    class(ConnectionsType) :: this
    integer(I4B), intent(in) :: nrsize
    integer(I4B), intent(in) :: nodesuser
    integer(I4B), dimension(:), intent(in) :: nodereduced
    integer(I4B), dimension(:), intent(in) :: nodeuser
    ! -- local
    integer(I4B) :: n, nr, ipos
! ------------------------------------------------------------------------------
    !
    ! -- If reduced system, then need to build iausr and jausr, otherwise point
    !    them to ia and ja.
    if(nrsize > 0) then
      !
      ! -- Create the iausr array of size nodesuser + 1.  For excluded cells,
      !    iausr(n) and iausr(n + 1) should be equal to indicate no connections.
      call mem_reallocate(this%iausr, nodesuser+1, 'IAUSR', this%memoryPath)
      this%iausr(nodesuser + 1) = this%ia(this%nodes + 1)
      do n = nodesuser, 1, -1
        nr = nodereduced(n)
        if(nr < 1) then
          this%iausr(n) = this%iausr(n + 1)
        else
          this%iausr(n) = this%ia(nr)
        endif
      enddo
      !
      ! -- Create the jausr array, which is the same size as ja, but it
      !    contains user node numbers instead of reduced node numbers
      call mem_reallocate(this%jausr, this%nja, 'JAUSR', this%memoryPath)
      do ipos = 1, this%nja
        nr = this%ja(ipos)
        n = nodeuser(nr)
        this%jausr(ipos) = n
      enddo
    else
      ! -- iausr and jausr will be pointers
      call mem_deallocate(this%iausr)
      call mem_deallocate(this%jausr)
      call mem_setptr(this%iausr, 'IA', this%memoryPath)
      call mem_setptr(this%jausr, 'JA', this%memoryPath)
    endif
    !
    ! -- Return
    return
  end subroutine iajausr

  function getjaindex(this,node1,node2)
! ******************************************************************************
! Get the index in the JA array corresponding to the connection between
! two nodes of interest.  Node1 is used as the index in the IA array, and
! IA(Node1) is the row index in the (nodes x nodes) matrix represented by
! the compressed sparse row format.
!
!   -1 is returned if either node number is invalid.
!    0 is returned if the two nodes are not connected.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    integer(I4B) :: getjaindex
    ! -- dummy
    class(ConnectionsType) :: this
    integer(I4B), intent(in)       :: node1, node2 ! nodes of interest
    ! -- local
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    ! -- error checking
    if (node1<1 .or. node1>this%nodes .or. node2<1 .or. node2>this%nodes) then
      getjaindex = -1  ! indicates error (an invalid node number)
      return
    endif
    !
    ! -- If node1==node2, just return the position for the diagonal.
    if (node1==node2) then
      getjaindex = this%ia(node1)
      return
    endif
    !
    ! -- Look for connection among nonzero elements defined for row node1.
    do i=this%ia(node1)+1,this%ia(node1+1)-1
      if (this%ja(i)==node2) then
        getjaindex = i
        return
      endif
    enddo
    !
    ! -- If execution reaches here, no connection exists
    !    between nodes of interest.
    getjaindex = 0  ! indicates no connection exists
    return
  end function getjaindex

  subroutine fillisym(neq, nja, ia, ja, isym)
! ******************************************************************************
! fillisym -- Private function to fill the isym array
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    integer(I4B),intent(in) :: neq
    integer(I4B),intent(in) :: nja
    integer(I4B),intent(inout),dimension(nja) :: isym
    ! -- local
    integer(I4B),intent(in),dimension(neq+1) :: ia
    integer(I4B),intent(in),dimension(nja) :: ja
    integer(I4B) :: n, m, ii, jj
! ------------------------------------------------------------------------------
    !
    do n=1, neq
      do ii = ia(n), ia(n + 1) - 1
        m = ja(ii)
        if(m /= n) then
          isym(ii) = 0
          search: do jj = ia(m), ia(m + 1) - 1
            if(ja(jj) == n) then
              isym(ii) = jj
              exit search
            endif
          enddo search
        else
          isym(ii) = ii
        endif
      enddo
    enddo
    !
    ! -- Return
    return
  end subroutine fillisym

  subroutine filljas(neq, nja, ia, ja, isym, jas)
! ******************************************************************************
! fillisym -- Private function to fill the jas array
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    integer(I4B),intent(in) :: neq
    integer(I4B),intent(in) :: nja
    integer(I4B),intent(in),dimension(neq+1) :: ia
    integer(I4B),intent(in),dimension(nja) :: ja
    integer(I4B),intent(in),dimension(nja) :: isym
    integer(I4B),intent(inout),dimension(nja) :: jas
    ! -- local
    integer(I4B) :: n, m, ii, ipos
! ------------------------------------------------------------------------------
    !
    ! -- set diagonal to zero and fill upper
    ipos = 1
    do n = 1, neq
      jas(ia(n)) = 0
      do ii = ia(n) + 1, ia(n + 1) - 1
        m = ja(ii)
        if(m > n) then
          jas(ii) = ipos
          ipos = ipos + 1
        endif
      enddo
    enddo
    !
    ! -- fill lower
    do n = 1, neq
      do ii = ia(n), ia(n + 1) - 1
        m = ja(ii)
        if(m < n) then
          jas(ii) = jas(isym(ii))
        endif
      enddo
    enddo
    !
    ! -- Return
    return
  end subroutine filljas


  subroutine vertexconnect(nodes, nrsize, maxnnz, nlay, ncpl, sparse,          &
                           vertcellspm, cell1, cell2, nodereduced)
! ******************************************************************************
! vertexconnect -- routine to make cell connections from vertices
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use SparseModule, only: sparsematrix
    use InputOutputModule, only: get_node
    use DisvGeom, only: DisvGeomType
    ! -- dummy
    integer(I4B), intent(in) :: nodes
    integer(I4B), intent(in) :: nrsize
    integer(I4B), intent(in) :: maxnnz
    integer(I4B), intent(in) :: nlay
    integer(I4B), intent(in) :: ncpl
    type(SparseMatrix), intent(inout) :: sparse
    type(SparseMatrix), intent(inout) :: vertcellspm
    integer(I4B), dimension(:), intent(in) :: nodereduced
    type(DisvGeomType), intent(inout) :: cell1, cell2
    ! -- local
    integer(I4B), dimension(:), allocatable :: rowmaxnnz
    integer(I4B) :: i, j, k, kk, nr, mr, j1, j2, icol1, icol2, nvert
! ------------------------------------------------------------------------------
    !
    ! -- Allocate and fill the ia and ja arrays
    allocate(rowmaxnnz(nodes))
    do i = 1, nodes
      rowmaxnnz(i) = maxnnz
    enddo
    call sparse%init(nodes, nodes, rowmaxnnz)
    deallocate(rowmaxnnz)
    do k = 1, nlay
      do j = 1, ncpl
        !
        ! -- Find the reduced node number and then cycle if the
        !    node is always inactive
        nr = get_node(k, 1, j, nlay, 1, ncpl)
        if(nrsize > 0) nr = nodereduced(nr)
        if(nr <= 0) cycle
        !
        ! -- Process diagonal
        call sparse%addconnection(nr, nr, 1)
        !
        ! -- Up direction
        if(k > 1) then
          do kk = k - 1, 1, -1
            mr = get_node(kk, 1, j, nlay, 1, ncpl)
            if(nrsize > 0) mr = nodereduced(mr)
            if(mr >= 0) exit
          enddo
          if(mr > 0) then
            call sparse%addconnection(nr, mr, 1)
          endif
        endif
        !
        ! -- Down direction
        if(k < nlay) then
          do kk = k + 1, nlay
            mr = get_node(kk, 1, j, nlay, 1, ncpl)
            if(nrsize > 0) mr = nodereduced(mr)
            if(mr >= 0) exit
          enddo
          if(mr > 0) then
            call sparse%addconnection(nr, mr, 1)
          endif
        endif
      enddo
    enddo
    !
    ! -- Go through each vertex and connect up all the cells that use
    !    this vertex in their definition and share an edge.
    nvert = vertcellspm%nrow
    do i = 1, nvert
      do icol1 = 1, vertcellspm%row(i)%nnz
        j1 = vertcellspm%row(i)%icolarray(icol1)
        do k = 1, nlay
          nr = get_node(k, 1, j1, nlay, 1, ncpl)
          if(nrsize > 0) nr = nodereduced(nr)
          if(nr <= 0) cycle
          call cell1%set_nodered(nr)
          do icol2 = 1, vertcellspm%row(i)%nnz
            j2 = vertcellspm%row(i)%icolarray(icol2)
            if(j1 == j2) cycle
            mr = get_node(k, 1, j2, nlay, 1, ncpl)
            if(nrsize > 0) mr = nodereduced(mr)
            if(mr <= 0) cycle
            call cell2%set_nodered(mr)
            if(cell1%shares_edge(cell2)) then
              call sparse%addconnection(nr, mr, 1)
            endif
          enddo
        enddo        
      enddo
    enddo
    !
    ! -- return
    return
  end subroutine vertexconnect
  
  subroutine set_mask(this, ipos, maskval)
! ******************************************************************************
! set_mask -- routine to set a value in the mask array 
! (which has the same shape as this%ja)
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------ 
    use MemoryManagerModule, only: mem_allocate
    class(ConnectionsType) :: this
    integer(I4B), intent(in) :: ipos
    integer(I4B), intent(in) :: maskval
    ! local
    integer(I4B) :: i
! ------------------------------------------------------------------------------ 
    !
    ! if we still point to this%ja, we first need to allocate space
    if (associated(this%mask, this%ja)) then
      call mem_allocate(this%mask, this%nja, 'MASK', this%memoryPath)
      ! and initialize with unmasked
      do i = 1, this%nja
        this%mask(i) = 1 
      end do
    end if
    !
    ! -- set the mask value
    this%mask(ipos) = maskVal
    !
    ! -- return
    return
  end subroutine set_mask
                           
  subroutine iac_to_ia(ia)
! ******************************************************************************
! iac_to_ia -- convert an iac array into an ia array
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------ 
    ! -- dummy
    integer(I4B), dimension(:), contiguous, intent(inout) :: ia
    ! -- local
    integer(I4B) :: n, nodes
! ------------------------------------------------------------------------------ 
    !
    ! -- Convert iac to ia
    nodes = size(ia) - 1
    do n = 2, nodes + 1
      ia(n) = ia(n) + ia(n-1)
    enddo
    do n = nodes + 1, 2, -1
      ia(n) = ia(n - 1) + 1
    enddo
    ia(1) = 1
    !
    ! -- return
    return
  end subroutine iac_to_ia

end module ConnectionsModule
