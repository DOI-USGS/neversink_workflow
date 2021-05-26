module SfrModule
  !
  use KindModule, only: DP, I4B
  use ConstantsModule, only: LINELENGTH, LENBOUNDNAME, LENTIMESERIESNAME,        &
                             DZERO, DPREC, DEM30, DEM6, DEM5, DEM4, DEM2,        &
                             DHALF, DP6, DTWOTHIRDS, DP7, DP9, DP99, DP999,      &
                             DONE, D1P1, DFIVETHIRDS, DTWO, DPI, DEIGHT,         &
                             DHUNDRED, DEP20,                                    &
                             NAMEDBOUNDFLAG, LENBOUNDNAME, LENFTYPE,             &
                             LENPACKAGENAME, LENPAKLOC, MAXCHARLEN,              &
                             DHNOFLO, DHDRY, DNODATA,                            &
                             TABLEFT, TABCENTER, TABRIGHT                
  use SmoothingModule,  only: sQuadraticSaturation, sQSaturation,                &
                              sQuadraticSaturationDerivative,                    &
                              sQSaturationDerivative,                            &
                              sCubicSaturation, sChSmooth
  use BndModule, only: BndType
  use BudgetObjectModule, only: BudgetObjectType, budgetobject_cr
  use TableModule, only: TableType, table_cr
  use ObserveModule, only: ObserveType
  use ObsModule, only: ObsType
  use InputOutputModule, only: get_node, URWORD, extract_idnum_or_bndname
  use BaseDisModule, only: DisBaseType
  use SimModule, only: count_errors, store_error, store_error_unit,              &
                       store_warning, ustop
  use SimVariablesModule, only: errmsg, warnmsg
  use GenericUtilitiesModule, only: sim_message
  !use ArrayHandlersModule, only: ExpandArray
  use BlockParserModule,   only: BlockParserType
  !
  implicit none
  !
  character(len=LENFTYPE)       :: ftype = 'SFR'
  character(len=LENPACKAGENAME) :: text  = '             SFR'
  !
  private
  public :: sfr_create
  public :: SfrType
  !
  type, extends(BndType) :: SfrType
    ! -- scalars
    ! -- for budgets
    ! -- characters
    character(len=16), dimension(:), pointer, contiguous :: csfrbudget => NULL()
    character(len=16), dimension(:), pointer, contiguous :: cauxcbc => NULL()
    character(len=LENBOUNDNAME), dimension(:), pointer,                         &
                                 contiguous :: sfrname => null()
    ! -- integers
    integer(I4B), pointer :: iprhed => null()
    integer(I4B), pointer :: istageout => null()
    integer(I4B), pointer :: ibudgetout => null()
    integer(I4B), pointer :: ipakcsv => null()
    integer(I4B), pointer :: idiversions => null()
    integer(I4B), pointer :: nconn => NULL()
    integer(I4B), pointer :: maxsfrpicard => NULL()
    integer(I4B), pointer :: maxsfrit => NULL()
    integer(I4B), pointer :: bditems => NULL()
    integer(I4B), pointer :: cbcauxitems => NULL()
    integer(I4B), pointer :: icheck => NULL()
    integer(I4B), pointer :: iconvchk => NULL()
    integer(I4B), pointer :: gwfiss => NULL()
    integer(I4B), pointer :: ianynone => null()                                   !< number of reaches with 'none' connection
    ! -- double precision
    real(DP), pointer :: unitconv => NULL()
    real(DP), pointer :: dmaxchg => NULL()
    real(DP), pointer :: deps => NULL()
    ! -- integer vectors
    integer(I4B), dimension(:), pointer, contiguous :: ia => null()
    integer(I4B), dimension(:), pointer, contiguous :: ja => null()
    ! -- double precision output vectors
    real(DP), dimension(:), pointer, contiguous :: qoutflow => null()
    real(DP), dimension(:), pointer, contiguous :: qextoutflow => null()
    real(DP), dimension(:), pointer, contiguous :: qauxcbc => null()
    real(DP), dimension(:), pointer, contiguous :: dbuff => null()
    !
    ! -- sfr budget object
    type(BudgetObjectType), pointer :: budobj => null()
    !
    ! -- sfr table objects
    type(TableType), pointer :: stagetab => null()
    type(TableType), pointer :: pakcsvtab => null()
    !
    ! -- moved from SfrDataType
    integer(I4B), dimension(:), pointer, contiguous :: iboundpak => null()
    integer(I4B), dimension(:), pointer, contiguous :: igwfnode => null()
    integer(I4B), dimension(:), pointer, contiguous :: igwftopnode => null()
    real(DP), dimension(:), pointer, contiguous :: length => null()
    real(DP), dimension(:), pointer, contiguous :: width => null()
    real(DP), dimension(:), pointer, contiguous :: strtop => null()
    real(DP), dimension(:), pointer, contiguous :: bthick => null()
    real(DP), dimension(:), pointer, contiguous :: hk => null()
    real(DP), dimension(:), pointer, contiguous :: slope => null()
    integer(I4B), dimension(:), pointer, contiguous :: nconnreach => null()
    real(DP), dimension(:), pointer, contiguous :: ustrf => null()
    real(DP), dimension(:), pointer, contiguous :: ftotnd => null()
    integer(I4B), dimension(:), pointer, contiguous :: ndiv => null()
    real(DP), dimension(:), pointer, contiguous :: usflow => null()
    real(DP), dimension(:), pointer, contiguous :: dsflow => null()
    real(DP), dimension(:), pointer, contiguous :: depth => null()
    real(DP), dimension(:), pointer, contiguous :: stage => null()
    real(DP), dimension(:), pointer, contiguous :: gwflow => null()
    real(DP), dimension(:), pointer, contiguous :: simevap => null()
    real(DP), dimension(:), pointer, contiguous :: simrunoff => null()
    real(DP), dimension(:), pointer, contiguous :: stage0 => null()
    real(DP), dimension(:), pointer, contiguous :: usflow0 => null()
    ! -- connection data
    integer(I4B), dimension(:), pointer, contiguous :: idir => null()
    integer(I4B), dimension(:), pointer, contiguous :: idiv => null()
    real(DP), dimension(:), pointer, contiguous :: qconn => null()
    ! -- boundary data
    real(DP), dimension(:), pointer, contiguous :: rough => null()
    real(DP), dimension(:), pointer, contiguous :: rain => null()
    real(DP), dimension(:), pointer, contiguous :: evap => null()
    real(DP), dimension(:), pointer, contiguous :: inflow => null()
    real(DP), dimension(:), pointer, contiguous :: runoff => null()
    real(DP), dimension(:), pointer, contiguous :: sstage => null()
    ! -- reach aux variables
    real(DP), dimension(:,:), pointer, contiguous :: rauxvar => null()
    ! -- diversion data
    integer(I4B), dimension(:), pointer, contiguous :: iadiv => null()
    integer(I4B), dimension(:), pointer, contiguous :: divreach => null()
    character (len=10), dimension(:), pointer, contiguous :: divcprior => null()
    real(DP), dimension(:), pointer, contiguous :: divflow => null()
    real(DP), dimension(:), pointer, contiguous :: divq => null()
    !
    ! -- density variables
    integer(I4B), pointer :: idense
    real(DP), dimension(:, :), pointer, contiguous  :: denseterms => null()
    !
    ! -- type bound procedures
    contains
    procedure :: sfr_allocate_scalars
    procedure :: sfr_allocate_arrays
    procedure :: bnd_options => sfr_options
    procedure :: read_dimensions => sfr_read_dimensions
    procedure :: set_pointers => sfr_set_pointers
    procedure :: bnd_ar => sfr_ar
    procedure :: bnd_rp => sfr_rp
    procedure :: bnd_ad => sfr_ad
    procedure :: bnd_cf => sfr_cf
    procedure :: bnd_fc => sfr_fc
    procedure :: bnd_fn => sfr_fn
    procedure :: bnd_cc => sfr_cc
    procedure :: bnd_bd => sfr_bd
    procedure :: bnd_ot => sfr_ot
    procedure :: bnd_da => sfr_da
    procedure :: define_listlabel
    ! -- methods for observations
    procedure, public :: bnd_obs_supported => sfr_obs_supported
    procedure, public :: bnd_df_obs => sfr_df_obs
    procedure, public :: bnd_rp_obs => sfr_rp_obs
    procedure, private :: sfr_bd_obs
    ! -- private procedures
    procedure, private :: sfr_set_stressperiod
    procedure, private :: sfr_solve
    procedure, private :: sfr_update_flows
    procedure, private :: sfr_calc_qgwf
    procedure, private :: sfr_calc_cond
    procedure, private :: sfr_calc_qman
    procedure, private :: sfr_calc_qd
    procedure, private :: sfr_calc_qsource
    procedure, private :: sfr_calc_div
    ! -- geometry 
    procedure, private :: area_wet
    procedure, private :: perimeter_wet
    procedure, private :: surface_area
    procedure, private :: surface_area_wet
    procedure, private :: top_width_wet
    ! -- reading
    procedure, private :: sfr_read_packagedata
    procedure, private :: sfr_read_connectiondata
    procedure, private :: sfr_read_diversions
    ! -- calculations
    procedure, private :: sfr_rectch_depth
    ! -- error checking
    procedure, private :: sfr_check_reaches
    procedure, private :: sfr_check_connections
    procedure, private :: sfr_check_diversions
    procedure, private :: sfr_check_ustrf
    ! -- budget
    procedure, private :: sfr_setup_budobj
    procedure, private :: sfr_fill_budobj
    ! -- table
    procedure, private :: sfr_setup_tableobj
    ! -- density
    procedure :: sfr_activate_density
    procedure, private :: sfr_calculate_density_exchange
  end type SfrType

contains

  subroutine sfr_create(packobj, id, ibcnum, inunit, iout, namemodel, pakname)
! ******************************************************************************
! sfr_create -- Create a New Streamflow Routing Package
! Subroutine: (1) create new-style package
!             (2) point bndobj to the new package
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use MemoryHelperModule, only: create_mem_path
    ! -- dummy
    class(BndType), pointer :: packobj
    integer(I4B),intent(in) :: id
    integer(I4B),intent(in) :: ibcnum
    integer(I4B),intent(in) :: inunit
    integer(I4B),intent(in) :: iout
    character(len=*), intent(in) :: namemodel
    character(len=*), intent(in) :: pakname
    ! -- local
    type(SfrType), pointer :: sfrobj
! ------------------------------------------------------------------------------
    !
    ! -- allocate the object and assign values to object variables
    allocate(sfrobj)
    packobj => sfrobj
    !
    ! -- create name and memory path
    call packobj%set_names(ibcnum, namemodel, pakname, ftype)
    packobj%text = text
    !
    ! -- allocate scalars
    call sfrobj%sfr_allocate_scalars()
    !
    ! -- initialize package
    call packobj%pack_initialize()

    packobj%inunit = inunit
    packobj%iout = iout
    packobj%id = id
    packobj%ibcnum = ibcnum
    packobj%ncolbnd = 4
    packobj%iscloc = 0  ! not supported
    packobj%ictMemPath = create_mem_path(namemodel,'NPF')
    !
    ! -- return
    return
  end subroutine sfr_create

  subroutine sfr_allocate_scalars(this)
! ******************************************************************************
! allocate_scalars -- allocate scalar members
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use MemoryManagerModule, only: mem_allocate, mem_setptr
    use MemoryHelperModule, only: create_mem_path
    ! -- dummy
    class(SfrType),   intent(inout) :: this
! ------------------------------------------------------------------------------
    !
    ! -- call standard BndType allocate scalars
    call this%BndType%allocate_scalars()
    !
    ! -- allocate the object and assign values to object variables
    call mem_allocate(this%iprhed, 'IPRHED', this%memoryPath)
    call mem_allocate(this%istageout, 'ISTAGEOUT', this%memoryPath)
    call mem_allocate(this%ibudgetout, 'IBUDGETOUT', this%memoryPath)
    call mem_allocate(this%ipakcsv, 'IPAKCSV', this%memoryPath)
    call mem_allocate(this%idiversions, 'IDIVERSIONS', this%memoryPath)
    call mem_allocate(this%maxsfrpicard, 'MAXSFRPICARD', this%memoryPath)
    call mem_allocate(this%maxsfrit, 'MAXSFRIT', this%memoryPath)
    call mem_allocate(this%bditems, 'BDITEMS', this%memoryPath)
    call mem_allocate(this%cbcauxitems, 'CBCAUXITEMS', this%memoryPath)
    call mem_allocate(this%unitconv, 'UNITCONV', this%memoryPath)
    call mem_allocate(this%dmaxchg, 'DMAXCHG', this%memoryPath)
    call mem_allocate(this%deps, 'DEPS', this%memoryPath)
    call mem_allocate(this%nconn, 'NCONN', this%memoryPath)
    call mem_allocate(this%icheck, 'ICHECK', this%memoryPath)
    call mem_allocate(this%iconvchk, 'ICONVCHK', this%memoryPath)
    call mem_allocate(this%idense, 'IDENSE', this%memoryPath)
    call mem_allocate(this%ianynone, 'IANYNONE', this%memoryPath)
    !
    ! -- set pointer to gwf iss
    call mem_setptr(this%gwfiss, 'ISS', create_mem_path(this%name_model))
    !
    ! -- Set values
    this%iprhed = 0
    this%istageout = 0
    this%ibudgetout = 0
    this%ipakcsv = 0
    this%idiversions = 0
    this%maxsfrpicard = 100
    this%maxsfrit = 100
    this%bditems = 8
    this%cbcauxitems = 1
    this%unitconv = DONE
    this%dmaxchg = DEM5
    this%deps = DP999 * this%dmaxchg
    this%nconn = 0
    this%icheck = 1
    this%iconvchk = 1
    this%idense = 0
    this%ianynone = 0
    !
    ! -- return
    return
  end subroutine sfr_allocate_scalars

  subroutine sfr_allocate_arrays(this)
! ******************************************************************************
! allocate_scalars -- allocate scalar members
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_allocate
    ! -- dummy
    class(SfrType),   intent(inout) :: this
    ! -- local
    integer(I4B) :: i
    integer(I4B) :: j
! ------------------------------------------------------------------------------
    !
    ! -- call standard BndType allocate scalars
    !call this%BndType%allocate_arrays()
    !
    ! -- allocate character array for budget text
    allocate(this%csfrbudget(this%bditems))
    call mem_allocate(this%sfrname, LENBOUNDNAME, this%maxbound,                 &
                      'SFRNAME', this%memoryPath)
    !
    ! -- variables originally in SfrDataType
    call mem_allocate(this%iboundpak, this%maxbound, 'IBOUNDPAK', this%memoryPath)
    call mem_allocate(this%igwfnode, this%maxbound, 'IGWFNODE', this%memoryPath)
    call mem_allocate(this%igwftopnode, this%maxbound, 'IGWFTOPNODE', this%memoryPath)
    call mem_allocate(this%length, this%maxbound, 'LENGTH', this%memoryPath)
    call mem_allocate(this%width, this%maxbound, 'WIDTH', this%memoryPath)
    call mem_allocate(this%strtop, this%maxbound, 'STRTOP', this%memoryPath)
    call mem_allocate(this%bthick, this%maxbound, 'BTHICK', this%memoryPath)
    call mem_allocate(this%hk, this%maxbound, 'HK', this%memoryPath)
    call mem_allocate(this%slope, this%maxbound, 'SLOPE', this%memoryPath)
    call mem_allocate(this%nconnreach, this%maxbound, 'NCONNREACH', this%memoryPath)
    call mem_allocate(this%ustrf, this%maxbound, 'USTRF', this%memoryPath)
    call mem_allocate(this%ftotnd, this%maxbound, 'FTOTND', this%memoryPath)
    call mem_allocate(this%ndiv, this%maxbound, 'NDIV', this%memoryPath)
    call mem_allocate(this%usflow, this%maxbound, 'USFLOW', this%memoryPath)
    call mem_allocate(this%dsflow, this%maxbound, 'DSFLOW', this%memoryPath)
    call mem_allocate(this%depth, this%maxbound, 'DEPTH', this%memoryPath)
    call mem_allocate(this%stage, this%maxbound, 'STAGE', this%memoryPath)
    call mem_allocate(this%gwflow, this%maxbound, 'GWFLOW', this%memoryPath)
    call mem_allocate(this%simevap, this%maxbound, 'SIMEVAP', this%memoryPath)
    call mem_allocate(this%simrunoff, this%maxbound, 'SIMRUNOFF', this%memoryPath)
    call mem_allocate(this%stage0, this%maxbound, 'STAGE0', this%memoryPath)
    call mem_allocate(this%usflow0, this%maxbound, 'USFLOW0', this%memoryPath)
    !
    ! -- connection data
    call mem_allocate(this%ia, this%maxbound+1, 'IA', this%memoryPath)
    call mem_allocate(this%ja, 0, 'JA', this%memoryPath)
    call mem_allocate(this%idir, 0, 'IDIR', this%memoryPath)
    call mem_allocate(this%idiv, 0, 'IDIV', this%memoryPath)
    call mem_allocate(this%qconn, 0, 'QCONN', this%memoryPath)
    !
    ! -- boundary data
    call mem_allocate(this%rough, this%maxbound, 'ROUGH', this%memoryPath)
    call mem_allocate(this%rain, this%maxbound, 'RAIN', this%memoryPath)
    call mem_allocate(this%evap, this%maxbound, 'EVAP', this%memoryPath)
    call mem_allocate(this%inflow, this%maxbound, 'INFLOW', this%memoryPath)
    call mem_allocate(this%runoff, this%maxbound, 'RUNOFF', this%memoryPath)
    call mem_allocate(this%sstage, this%maxbound, 'SSTAGE', this%memoryPath)
    !
    ! -- aux variables
    call mem_allocate(this%rauxvar, this%naux, this%maxbound,                    &
                      'RAUXVAR', this%memoryPath)
    !
    ! -- diversion variables
    call mem_allocate(this%iadiv, this%maxbound+1, 'IADIV', this%memoryPath)
    call mem_allocate(this%divreach, 0, 'DIVREACH', this%memoryPath)
    call mem_allocate(this%divflow, 0, 'DIVFLOW', this%memoryPath)
    call mem_allocate(this%divq, 0, 'DIVQ', this%memoryPath)
    !
    ! -- initialize variables
    do i = 1, this%maxbound
      this%iboundpak(i) = 1
      this%igwfnode(i) = 0
      this%igwftopnode(i) = 0
      this%length(i) = DZERO
      this%width(i) = DZERO
      this%strtop(i) = DZERO
      this%bthick(i) = DZERO
      this%hk(i) = DZERO
      this%slope(i) = DZERO
      this%nconnreach(i) = 0
      this%ustrf(i) = DZERO
      this%ftotnd(i) = DZERO
      this%ndiv(i) = 0
      this%usflow(i) = DZERO
      this%dsflow(i) = DZERO
      this%depth(i) = DZERO
      this%stage(i) = DZERO
      this%gwflow(i) = DZERO
      this%simevap(i) = DZERO
      this%simrunoff(i) = DZERO
      this%stage0(i) = DZERO
      this%usflow0(i) = DZERO
      !
      ! -- boundary data
      this%rough(i) = DZERO
      this%rain(i) = DZERO
      this%evap(i) = DZERO
      this%inflow(i) = DZERO
      this%runoff(i) = DZERO
      this%sstage(i) = DZERO
      !
      ! -- aux variables
      do j = 1, this%naux
        this%rauxvar(j, i) = DZERO
      end do
    end do
    
    !
    !
    !-- fill csfrbudget
    this%csfrbudget(1) = '        RAINFALL'
    this%csfrbudget(2) = '     EVAPORATION'
    this%csfrbudget(3) = '          RUNOFF'
    this%csfrbudget(4) = '      EXT-INFLOW'
    this%csfrbudget(5) = '             GWF'
    this%csfrbudget(6) = '     EXT-OUTFLOW'
    this%csfrbudget(7) = '        FROM-MVR'
    this%csfrbudget(8) = '          TO-MVR'
    !
    ! -- allocate and initialize budget output data
    call mem_allocate(this%qoutflow, this%maxbound, 'QOUTFLOW', this%memoryPath)
    call mem_allocate(this%qextoutflow, this%maxbound, 'QEXTOUTFLOW', this%memoryPath)
    do i = 1, this%maxbound
      this%qoutflow(i) = DZERO
      this%qextoutflow(i) = DZERO
    end do
    !
    ! -- allocate and initialize dbuff
    if (this%istageout > 0) then
      call mem_allocate(this%dbuff, this%maxbound, 'DBUFF', this%memoryPath)
      do i = 1, this%maxbound
        this%dbuff(i) = DZERO
      end do
    else
      call mem_allocate(this%dbuff, 0, 'DBUFF', this%memoryPath)
    end if
    !
    ! -- allocate character array for budget text
    allocate(this%cauxcbc(this%cbcauxitems))
    !
    ! -- allocate and initialize qauxcbc
    call mem_allocate(this%qauxcbc, this%cbcauxitems, 'QAUXCBC', this%memoryPath)
    do i = 1, this%cbcauxitems
      this%qauxcbc(i) = DZERO
    end do
    !
    !-- fill cauxcbc
    this%cauxcbc(1) = 'FLOW-AREA       '
    !
    ! -- allocate denseterms to size 0
    call mem_allocate(this%denseterms, 3, 0, 'DENSETERMS', this%memoryPath)
    !
    ! -- return
    return
  end subroutine sfr_allocate_arrays

  subroutine sfr_read_dimensions(this)
! ******************************************************************************
! pak1read_dimensions -- Read the dimensions for this package
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use ConstantsModule, only: LINELENGTH
    use InputOutputModule, only: urword
    use SimModule, only: ustop, store_error, count_errors
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: keyword
    integer(I4B) :: ierr
    logical :: isfound, endOfBlock
    ! -- format
! ------------------------------------------------------------------------------
    !
    ! -- initialize dimensions to 0
    this%maxbound = 0
    !
    ! -- get dimensions block
    call this%parser%GetBlock('DIMENSIONS', isFound, ierr, &
                              supportOpenClose=.true.)
    !
    ! -- parse dimensions block if detected
    if (isfound) then
      write(this%iout,'(/1x,a)')                                                 &
        'PROCESSING ' // trim(adjustl(this%text)) // ' DIMENSIONS'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(keyword)
        select case (keyword)
          case ('NREACHES')
            this%maxbound = this%parser%GetInteger()
            write(this%iout,'(4x,a,i0)')'NREACHES = ', this%maxbound
          case default
            write(errmsg,'(2a)')                                                 &
              'Unknown ' // trim(this%text) // ' dimension: ', trim(keyword)
            call store_error(errmsg)
        end select
      end do
      write(this%iout,'(1x,a)')                                                  &
        'END OF ' // trim(adjustl(this%text)) // ' DIMENSIONS'
    else
      call store_error('Required dimensions block not found.')
    end if
    !
    ! -- verify dimensions were set
    if(this%maxbound < 1) then
      write(errmsg, '(a)')                                                       &
        'NREACHES was not specified or was specified incorrectly.'
      call store_error(errmsg)
    endif
    !
    ! -- write summary of error messages for block
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- Call define_listlabel to construct the list label that is written
    !    when PRINT_INPUT option is used.
    call this%define_listlabel()
    !
    ! -- Allocate arrays in package superclass
    call this%sfr_allocate_arrays()
    !
    ! -- read package data
    call this%sfr_read_packagedata()
    !
    ! -- read connection data
    call this%sfr_read_connectiondata()
    !
    ! -- read diversion data
    call this%sfr_read_diversions()
    !
    ! -- setup the budget object
    call this%sfr_setup_budobj()
    !
    ! -- setup the stage table object
    call this%sfr_setup_tableobj()
    !
    ! -- return
    return
  end subroutine sfr_read_dimensions

  subroutine sfr_options(this, option, found)
! ******************************************************************************
! rch_options -- set options specific to RchType
!
! rch_options overrides BndType%bnd_options
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use ConstantsModule, only: DZERO, MNORMAL
    use OpenSpecModule, only: access, form
    use SimModule, only: ustop, store_error
    use InputOutputModule, only: urword, getunit, openfile
    ! -- dummy
    class(SfrType),   intent(inout) :: this
    character(len=*), intent(inout) :: option
    logical,          intent(inout) :: found
    ! -- local
    real(DP) :: r
    character(len=MAXCHARLEN) :: fname, keyword
    ! -- formats
    character(len=*),parameter :: fmtunitconv = &
      "(4x, 'UNIT CONVERSION VALUE (',g0,') SPECIFIED.')"
    character(len=*),parameter :: fmtpicard = &
      "(4x, 'MAXIMUM SFR PICARD ITERATION VALUE (',i0,') SPECIFIED.')"
    character(len=*),parameter :: fmtiter = &
      "(4x, 'MAXIMUM SFR ITERATION VALUE (',i0,') SPECIFIED.')"
    character(len=*),parameter :: fmtdmaxchg = &
      "(4x, 'MAXIMUM DEPTH CHANGE VALUE (',g0,') SPECIFIED.')"
    character(len=*),parameter :: fmtsfrbin = &
      "(4x, 'SFR ', 1x, a, 1x, ' WILL BE SAVED TO FILE: ', a, /4x,               &
     &'OPENED ON UNIT: ', I7)"
! ------------------------------------------------------------------------------
    !
    ! -- Check for SFR options
    select case (option)
      case ('PRINT_STAGE')
        this%iprhed = 1
        write(this%iout,'(4x,a)') trim(adjustl(this%text))//                     &
          ' STAGES WILL BE PRINTED TO LISTING FILE.'
        found = .true.
      case('STAGE')
        call this%parser%GetStringCaps(keyword)
        if (keyword == 'FILEOUT') then
          call this%parser%GetString(fname)
          this%istageout = getunit()
          call openfile(this%istageout, this%iout, fname, 'DATA(BINARY)',        &
                       form, access, 'REPLACE', MNORMAL)
          write(this%iout,fmtsfrbin) 'STAGE', fname, this%istageout
          found = .true.
        else
          call store_error('Optional stage keyword must be followed by fileout.')
        end if
      case('BUDGET')
        call this%parser%GetStringCaps(keyword)
        if (keyword == 'FILEOUT') then
          call this%parser%GetString(fname)
          this%ibudgetout = getunit()
          call openfile(this%ibudgetout, this%iout, fname, 'DATA(BINARY)',       &
                        form, access, 'REPLACE', MNORMAL)
          write(this%iout,fmtsfrbin) 'BUDGET', fname, this%ibudgetout
          found = .true.
        else
          call store_error('Optional budget keyword must be ' //                 &
                           'followed by fileout.')
        end if
      case('PACKAGE_CONVERGENCE')
        call this%parser%GetStringCaps(keyword)
        if (keyword == 'FILEOUT') then
          call this%parser%GetString(fname)
          this%ipakcsv = getunit()
          call openfile(this%ipakcsv, this%iout, fname, 'CSV',                   &
                        filstat_opt='REPLACE', mode_opt=MNORMAL)
          write(this%iout,fmtsfrbin) 'PACKAGE_CONVERGENCE', fname, this%ipakcsv
          found = .true.
        else
          call store_error('Optional package_convergence keyword must be ' //    &
                           'followed by fileout.')
        end if
      case('UNIT_CONVERSION')
        this%unitconv = this%parser%GetDouble()
        write(this%iout, fmtunitconv) this%unitconv
        found = .true.
      case('MAXIMUM_PICARD_ITERATIONS')
        this%maxsfrpicard = this%parser%GetInteger()
        write(this%iout, fmtpicard) this%maxsfrpicard
        found = .true.
      case('MAXIMUM_ITERATIONS')
        this%maxsfrit = this%parser%GetInteger()
        write(this%iout, fmtiter) this%maxsfrit
        found = .true.
      case('MAXIMUM_DEPTH_CHANGE')
        r = this%parser%GetDouble()
        this%dmaxchg = r
        this%deps = DP999 * r
        write(this%iout, fmtdmaxchg) this%dmaxchg
        found = .true.
      case('MOVER')
        this%imover = 1
        write(this%iout, '(4x,A)') 'MOVER OPTION ENABLED'
        found = .true.
      !
      ! -- right now these are options that are only available in the
      !    development version and are not included in the documentation.
      !    These options are only available when IDEVELOPMODE in
      !    constants module is set to 1
      case('DEV_NO_CHECK')
        call this%parser%DevOpt()
        this%icheck = 0
        write(this%iout, '(4x,A)') 'SFR CHECKS OF REACH GEOMETRY ' //         &
                                   'RELATIVE TO MODEL GRID AND ' //           &
                                   'REASONABLE PARAMETERS WILL NOT ' //       &
                                   'BE PERFORMED.'
        found = .true.
      case('DEV_NO_FINAL_CHECK')
        call this%parser%DevOpt()
        this%iconvchk = 0
        write(this%iout, '(4x,a)')                                             &
     &    'A FINAL CONVERGENCE CHECK OF THE CHANGE IN STREAM FLOW ROUTING ' // &
     &    'STAGES AND FLOWS WILL NOT BE MADE'
        found = .true.
      !
      ! -- no valid options found
      case default
        !
        ! -- No options found
        found = .false.
    end select
    !
    ! -- return
    return
  end subroutine sfr_options

  subroutine sfr_ar(this)
  ! ******************************************************************************
  ! sfr_ar -- Allocate and Read
  ! Subroutine: (1) create new-style package
  !             (2) point bndobj to the new package
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    use SimModule, only: ustop, count_errors
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    integer(I4B) :: n, ierr
    ! -- format
  ! ------------------------------------------------------------------------------
    !
    call this%obs%obs_ar()
    !
    ! -- call standard BndType allocate scalars
    call this%BndType%allocate_arrays()
    !
    ! -- set boundname for each connection
    if (this%inamedbound /= 0) then
      do n = 1, this%maxbound
        this%boundname(n) = this%sfrname(n)
      end do
    endif
    !
    ! -- copy igwfnode into nodelist
    do n = 1, this%maxbound
      this%nodelist(n) = this%igwfnode(n)
    end do
    !
    ! -- check the sfr data
    call this%sfr_check_reaches()

    ! -- check the connection data
    call this%sfr_check_connections()

    ! -- check the diversion data
    if (this%idiversions /= 0) then
      call this%sfr_check_diversions()
    end if
    !
    ! -- terminate if errors were detected in any of the static sfr data
    ierr = count_errors()
    if (ierr > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- setup pakmvrobj
    if (this%imover /= 0) then
      allocate(this%pakmvrobj)
      call this%pakmvrobj%ar(this%maxbound, this%maxbound, this%memoryPath)
    endif
    !
    ! -- return
    return
  end subroutine sfr_ar

  subroutine sfr_read_packagedata(this)
  ! ******************************************************************************
  ! sfr_read_packagedata -- read package data for each reach
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    use ConstantsModule, only: LINELENGTH
    use SimModule, only: ustop, store_error, count_errors
    use TimeSeriesManagerModule, only: read_value_or_time_series_adv
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    character(len=LINELENGTH) :: text
    character(len=LINELENGTH) :: cellid
    character(len=LINELENGTH) :: keyword
    character (len=10) :: cnum
    character(len=LENBOUNDNAME) :: bndName
    character(len=LENBOUNDNAME) :: bndNameTemp
    character(len=LENBOUNDNAME) :: manningname
    character(len=LENBOUNDNAME) :: ustrfname
    character(len=50), dimension(:), allocatable :: caux
    integer(I4B) :: n, ierr, ival
    logical :: isfound, endOfBlock
    integer(I4B) :: i
    integer(I4B) :: ii
    integer(I4B) :: jj
    integer(I4B) :: iaux
    integer(I4B) :: nconzero
    integer, allocatable, dimension(:) :: nboundchk
    real(DP), pointer :: bndElem => null()
    ! -- format
  ! ------------------------------------------------------------------------------
    !
    ! -- allocate space for checking sfr reach data
    allocate(nboundchk(this%maxbound))
    do i = 1, this%maxbound
      nboundchk(i) = 0
    enddo 
    nconzero = 0
    !
    ! -- allocate local storage for aux variables
    if (this%naux > 0) then
      allocate(caux(this%naux))
    end if
    !
    ! -- read reach data
    call this%parser%GetBlock('PACKAGEDATA', isfound, ierr, &
                              supportOpenClose=.true.)
    !
    ! -- parse reaches block if detected
    if (isfound) then
      write(this%iout,'(/1x,a)')'PROCESSING '//trim(adjustl(this%text))// &
        ' PACKAGEDATA'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        ! -- read reach number
        n = this%parser%GetInteger()

        if (n < 1 .or. n > this%maxbound) then
          write(errmsg,'(a,1x,a,1x,i0)')                                         &
            'Reach number (rno) must be greater than 0 and less',                &
            'than or equal to', this%maxbound
          call store_error(errmsg)
          cycle
        end if

        ! -- increment nboundchk
        nboundchk(n) = nboundchk(n) + 1
        !
        ! -- get model node number
        call this%parser%GetCellid(this%dis%ndim, cellid, flag_string=.true.)
        this%igwfnode(n) = this%dis%noder_from_cellid(cellid, &
                           this%inunit, this%iout, flag_string=.true.)
        this%igwftopnode(n) = this%igwfnode(n)
        !
        ! -- read the cellid string and determine if 'none' is specified
        if (this%igwfnode(n) < 1) then
          call this%parser%GetStringCaps(keyword)
          this%ianynone = this%ianynone + 1
          if (keyword /= 'NONE') then
            write(cnum, '(i0)') n
            errmsg = 'Cell ID (' // trim(cellid) //                              &
                     ') for unconnected reach ' //  trim(cnum) //                &
                     ' must be NONE'
            call store_error(errmsg)
          end if
        end if
        ! -- get reach length
        this%length(n) = this%parser%GetDouble()
        ! -- get reach width
        this%width(n) = this%parser%GetDouble()
        ! -- get reach slope
        this%slope(n) = this%parser%GetDouble()
        ! -- get reach stream bottom
        this%strtop(n) = this%parser%GetDouble()
        ! -- get reach bed thickness
        this%bthick(n) = this%parser%GetDouble()
        ! -- get reach bed hk
        this%hk(n) = this%parser%GetDouble()
        ! -- get reach roughness
        call this%parser%GetStringCaps(manningname)
        ! -- get number of connections for reach
        ival = this%parser%GetInteger()
        this%nconnreach(n) = ival
        this%nconn = this%nconn + ival
        if (ival < 0) then
          write(errmsg, '(a,1x,i0,1x,a,i0,a)')                                   &
            'NCON for reach', n,                                                 &
            'must be greater than or equal to 0 (', ival, ').'
          call store_error(errmsg)
        else if (ival == 0) then
          nconzero = nconzero + 1
        end if
        ! -- get upstream fraction for reach
        call this%parser%GetString(ustrfname)
        ! -- get number of diversions for reach
        ival = this%parser%GetInteger()
        this%ndiv(n) = ival
        if (ival > 0) then
          this%idiversions = 1
        else if (ival < 0) then
          ival = 0
        end if

        ! -- get aux data
        do iaux = 1, this%naux
          call this%parser%GetString(caux(iaux))
        end do

        ! -- set default bndName
        write(cnum,'(i10.10)') n
        bndName = 'Reach' // cnum

        ! -- get reach name
        if (this%inamedbound /= 0) then
          call this%parser%GetStringCaps(bndNameTemp)
          if (bndNameTemp /= '') then
            bndName = bndNameTemp
          endif
          !this%boundname(n) = bndName
        end if
        this%sfrname(n) = bndName
        !
        ! -- set Mannings
        text = manningname
        jj = 1 !for 'ROUGH'
        bndElem => this%rough(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                            'BND', this%tsManager, this%iprpak,  &
                                            'MANNING')
        !
        ! -- set upstream fraction
        text = ustrfname
        jj = 1  ! For 'USTRF'
        bndElem => this%ustrf(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'USTRF')
        !
        ! -- get aux data
        do jj = 1, this%naux
          text = caux(jj)
          ii = n
          bndElem => this%rauxvar(jj, ii)
          call read_value_or_time_series_adv(text, ii, jj, bndElem, this%packName,   &
                                             'AUX', this%tsManager, this%iprpak, &
                                             this%auxname(jj))
        end do
        !
        ! -- initialize sstage to the top of the reach
        !    this value would be used by simple routing reaches
        !    on kper = 1 and kstp = 1 if a stage is not specified
        !    on the status line for the reach
        this%sstage(n) = this%strtop(n)

      end do
      write(this%iout,'(1x,a)')                                                  &
        'END OF '//trim(adjustl(this%text))//' PACKAGEDATA'
    else
      call store_error('REQUIRED PACKAGEDATA BLOCK NOT FOUND.')
    end if
    !
    ! -- Check to make sure that every reach is specified and that no reach
    !    is specified more than once.
    do i = 1, this%maxbound
      if (nboundchk(i) == 0) then
        write(errmsg, '(a,i0,1x,a)')                                             &
          'Information for reach ', i, 'not specified in packagedata block.'
        call store_error(errmsg)
      else if (nboundchk(i) > 1) then
        write(errmsg, '(a,1x,i0,1x,a,1x,i0)')                                    &
          'Reach information specified', nboundchk(i), 'times for reach', i
        call store_error(errmsg)
      endif
    end do
    deallocate(nboundchk)
    !
    ! -- Submit warning message if any reach has zero connections
    if (nconzero > 0) then
      write(warnmsg, '(a,1x,a,1x,a,1x,i0,1x, a)')                              &
        'SFR Package', trim(this%packName),                                    &
        'has', nconzero, 'reach(es) with zero connections.'
      call store_warning(warnmsg)
    endif
    !
    ! -- terminate if errors encountered in reach block
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- deallocate local storage for aux variables
    if (this%naux > 0) then
      deallocate(caux)
    end if
    !
    ! -- return
    return
  end subroutine sfr_read_packagedata

  subroutine sfr_read_connectiondata(this)
  ! ******************************************************************************
  ! sfr_read_connectiondata -- read connection data for each reach
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    use ConstantsModule, only: LINELENGTH
    use MemoryManagerModule, only: mem_reallocate
    use SimModule, only: ustop, store_error, count_errors
    use SparseModule, only: sparsematrix
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: line
    logical :: isfound
    logical :: endOfBlock
    integer(I4B) :: n
    integer(I4B) :: i 
    integer(I4B) :: j
    integer(I4B) :: jj
    integer(I4B) :: jcol
    integer(I4B) :: jcol2
    integer(I4B) :: nja
    integer(I4B) :: ival
    integer(I4B) :: idir
    integer(I4B) :: ierr
    integer(I4B) :: nconnmax
    integer(I4B), dimension(:), pointer, contiguous :: rowmaxnnz => null()
    integer, allocatable, dimension(:) :: nboundchk
    integer, allocatable, dimension(:,:) :: iconndata
    type(sparsematrix), pointer :: sparse => null()
    ! -- format
  ! ------------------------------------------------------------------------------
    !
    ! -- allocate and initialize local variables for reach connections
    allocate(nboundchk(this%maxbound))
    do n = 1, this%maxbound
      nboundchk(n) = 0
    end do
    !
    ! -- calculate the number of non-zero entries (size of ja maxtrix)
    nja = 0
    nconnmax = 0
    allocate(rowmaxnnz(this%maxbound))
    do n = 1, this%maxbound
      ival = this%nconnreach(n)
      if (ival < 0) ival = 0
      rowmaxnnz(n) = ival + 1
      nja = nja + ival + 1
      if (ival > nconnmax) then
        nconnmax = ival
      end if
    end do 
    !
    ! -- reallocate connection data for package
    call mem_reallocate(this%ja, nja, 'JA', this%memoryPath)
    call mem_reallocate(this%idir, nja, 'IDIR', this%memoryPath)
    call mem_reallocate(this%idiv, nja, 'IDIV', this%memoryPath)
    call mem_reallocate(this%qconn, nja, 'QCONN', this%memoryPath)
    !
    ! -- initialize connection data
    do n = 1, nja
      this%idir(n) = 0
      this%idiv(n) = 0
      this%qconn(n) = DZERO
    end do
    !
    ! -- allocate space for iconndata
    allocate(iconndata(nconnmax, this%maxbound))
    !
    ! -- initialize iconndata
    do n = 1, this%maxbound
      do j = 1, nconnmax
        iconndata(j, n) = 0
      end do
    end do
    !
    ! -- allocate space for connectivity
    allocate(sparse)
    !
    ! -- set up sparse
    call sparse%init(this%maxbound, this%maxbound, rowmaxnnz)
    !
    ! -- read connection data
    call this%parser%GetBlock('CONNECTIONDATA', isfound, ierr, &
                              supportOpenClose=.true.)
    !
    ! -- parse reach connectivity block if detected
    if (isfound) then
      write(this%iout,'(/1x,a)')                                                 &
        'PROCESSING ' // trim(adjustl(this%text)) // ' CONNECTIONDATA'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        !
        ! -- get reach number
        n = this%parser%GetInteger()
        !
        ! -- check for error
        if (n < 1 .or. n > this%maxbound) then
          write(errmsg, '(a,1x,a,1x,i0)')                                        &
            'SFR reach in connectiondata block is less than one or greater',     &
            'than NREACHES:', n
          call store_error(errmsg)
          cycle
        endif
        !
        ! -- increment nboundchk
        nboundchk(n) = nboundchk(n) + 1
        !
        ! -- add diagonal connection for reach
        call sparse%addconnection(n, n, 1)
        !
        ! -- fill off diagonals
        do i = 1, this%nconnreach(n)
          !
          ! -- get connected reach
          ival = this%parser%GetInteger()
          !
          ! -- save connection data to temporary iconndata
          iconndata(i, n) = ival
          !
          ! -- determine idir
          if (ival < 0) then
            idir = -1
            ival = abs(ival)
          elseif (ival == 0) then
            call store_error('Missing or zero connection reach in line:')
            call store_error(line)
          else
            idir = 1
          end if
          if (ival > this%maxbound) then
            call store_error('Reach number exceeds NREACHES in line:')
            call store_error(line)
          endif
          !
          ! -- add connection to sparse
          call sparse%addconnection(n, ival, 1)
        end do
      end do
      
      write(this%iout,'(1x,a)')                                                  &
        'END OF ' // trim(adjustl(this%text)) // ' CONNECTIONDATA'
      
      do n = 1, this%maxbound
        !
        ! -- check for missing or duplicate sfr connections
        if (nboundchk(n) == 0) then
          write(errmsg,'(a,1x,i0)')                                              &
            'No connection data specified for reach', n
          call store_error(errmsg)
        else if (nboundchk(n) > 1) then
          write(errmsg,'(a,1x,i0,1x,a,1x,i0,1x,a)')                              &
            'Connection data for reach', n,                                      &
            'specified', nboundchk(n), 'times.'
          call store_error(errmsg)
        end if
      end do
    else
      call store_error('Required connectiondata block not found.')
    end if
    !
    ! -- terminate if errors encountered in connectiondata block
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- create ia and ja from sparse
    call sparse%filliaja(this%ia, this%ja, ierr, sort=.TRUE.)
    !
    ! -- test for error condition
    if (ierr /= 0) then
      write(errmsg, '(a,3(1x,a))')                                               &
        'Could not fill', trim(this%packName),                                   &
        'package IA and JA connection data.',                                    &
        'Check connectivity data in connectiondata block.'
      call store_error(errmsg)
    end if
    !
    ! -- fill flat connection storage
    do n = 1, this%maxbound
      do j = this%ia(n) + 1, this%ia(n+1) - 1
        jcol = this%ja(j)
        do jj = 1, this%nconnreach(n)
          jcol2 = iconndata(jj, n)
          if (abs(jcol2) == jcol) then
            idir = 1
            if (jcol2 < 0) then
              idir = -1
            end if
            this%idir(j) = idir
            exit
          end if
        end do
      end do
    end do
    !
    ! -- deallocate temporary local storage for reach connections
    deallocate(rowmaxnnz)
    deallocate(nboundchk)
    deallocate(iconndata)
    !
    ! -- destroy sparse
    call sparse%destroy()
    deallocate(sparse)
    !
    ! -- return
    return
  end subroutine sfr_read_connectiondata


  subroutine sfr_read_diversions(this)
  ! ******************************************************************************
  ! sfr_read_diversions -- read diversion data for each diversion
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    use ConstantsModule, only: LINELENGTH
    use MemoryManagerModule, only: mem_reallocate
    use SimModule, only: ustop, store_error, count_errors
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    character (len=10) :: cnum
    character (len=10) :: cval
    integer(I4B) :: j
    integer(I4B) :: n
    integer(I4B) :: ierr
    integer(I4B) :: ival
    integer(I4B) :: i0
    integer(I4B) :: ipos
    integer(I4B) :: jpos
    integer(I4B) :: ndiv
    integer(I4B) :: ndiversions
    integer(I4B) :: idivreach
    logical :: isfound, endOfBlock
    integer(I4B) :: idiv
    integer, allocatable, dimension(:) :: iachk
    integer, allocatable, dimension(:) :: nboundchk
    ! -- format
  ! ------------------------------------------------------------------------------
    !
    ! -- determine the total number of diversions and fill iadiv
    ndiversions = 0
    i0 = 1
    this%iadiv(1) = i0
    do n = 1, this%maxbound
      ndiversions = ndiversions + this%ndiv(n)
      i0 = i0 + this%ndiv(n)
      this%iadiv(n+1) = i0
    end do
    !
    ! -- reallocate memory for diversions
    if (ndiversions > 0) then
      call mem_reallocate(this%divreach, ndiversions, 'DIVREACH', this%memoryPath)
      allocate(this%divcprior(ndiversions))
      call mem_reallocate(this%divflow, ndiversions, 'DIVFLOW', this%memoryPath)
      call mem_reallocate(this%divq, ndiversions, 'DIVQ', this%memoryPath)
    end if
    !
    ! -- inititialize diversion flow
    do n = 1, ndiversions
      this%divflow(n) = DZERO
      this%divq(n) = DZERO
    end do
    !
    ! -- read diversions
    call this%parser%GetBlock('DIVERSIONS', isfound, ierr,                      &
                              supportOpenClose=.true.,                          &
                              blockRequired=.false.)
    !
    ! -- parse reach connectivity block if detected
    if (isfound) then
      if (this%idiversions /= 0) then
        write(this%iout,'(/1x,a)') 'PROCESSING ' // trim(adjustl(this%text)) // &
                                   ' DIVERSIONS'
        !
        ! -- allocate and initialize local variables for diversions
        ndiv = 0
        do n = 1, this%maxbound
          ndiv = ndiv + this%ndiv(n)
        end do
        allocate(iachk(this%maxbound+1))
        allocate(nboundchk(ndiv))
        iachk(1) = 1
        do n = 1, this%maxbound
          iachk(n+1) = iachk(n) + this%ndiv(n)
        end do
        do n = 1, ndiv
          nboundchk(n) = 0
        end do
        !
        ! -- read diversion data
        do
          call this%parser%GetNextLine(endOfBlock)
          if (endOfBlock) exit
          !
          ! -- get reach number
          n = this%parser%GetInteger()
          if (n < 1 .or. n > this%maxbound) then
            write(cnum, '(i0)') n
            errmsg = 'Reach number should be between 1 and ' //                  &
                      trim(cnum) // '.'
            call store_error(errmsg)
            cycle
          end if
          !
          ! -- make sure reach has at least one diversion
          if (this%ndiv(n) < 1) then
            write(cnum, '(i0)') n
            errmsg = 'Diversions cannot be specified ' //                        &
                     'for reach ' // trim(cnum)
            call store_error(errmsg)
            cycle
          end if
          !
          ! -- read diversion number
          ival = this%parser%GetInteger()
          if (ival < 1 .or. ival > this%ndiv(n)) then
            write(cnum, '(i0)') n
            errmsg = 'Reach  ' // trim(cnum)
            write(cnum, '(i0)') this%ndiv(n)
            errmsg = trim(errmsg) // ' diversion number should be between ' //   &
                     '1 and ' // trim(cnum) // '.'
            call store_error(errmsg)
            cycle
          end if
          
          ! -- increment nboundchk
          ipos = iachk(n) + ival - 1
          nboundchk(ipos) = nboundchk(ipos) + 1
          
          idiv = ival
          !
          ! -- get target reach for diversion
          ival = this%parser%GetInteger()
          if (ival < 1 .or. ival > this%maxbound) then
            write(cnum, '(i0)') ival
            errmsg = 'Diversion target reach number should be ' //               &
                     'between 1 and ' // trim(cnum) // '.'
            call store_error(errmsg)
            cycle
          end if
          idivreach = ival
          jpos = this%iadiv(n) + idiv - 1
          this%divreach(jpos) = idivreach
          !
          ! -- get cprior
          call this%parser%GetStringCaps(cval)
          ival = -1
          select case (cval)
            case('UPTO')
              ival = 0
            case('THRESHOLD')
              ival = -1
            case('FRACTION')
              ival = -2
            case('EXCESS')
              ival = -3
            case default
              errmsg = 'Invalid cprior type ' // trim(cval) // '.'
              call store_error(errmsg)
          end select
          !
          ! -- set cprior for diversion
          this%divcprior(jpos) = cval
        end do
        
        write(this%iout,'(1x,a)') 'END OF ' // trim(adjustl(this%text)) //       &
                                  ' DIVERSIONS'
        
        do n = 1, this%maxbound
          do j = 1, this%ndiv(n)
            ipos = iachk(n) + j - 1
            !
            ! -- check for missing or duplicate reach diversions
            if (nboundchk(ipos) == 0) then
              write(errmsg,'(a,1x,i0,1x,a,1x,i0)')                               &
                'No data specified for reach', n, 'diversion', j
              call store_error(errmsg)
            else if (nboundchk(ipos) > 1) then
              write(errmsg,'(a,1x,i0,1x,a,1x,i0,1x,a,1x,i0,1x,a)')               &
                'Data for reach', n, 'diversion', j,                             &
                'specified', nboundchk(ipos), 'times'
              call store_error(errmsg)
            end if
          end do
        end do
        !
        ! -- deallocate local variables
        deallocate(iachk)
        deallocate(nboundchk)
      else
        !
        ! -- error condition
        write(errmsg,'(a,1x,a)')                                                 &
          'A diversions block should not be',                                    &
          'specified if diversions are not specified.'
          call store_error(errmsg)
      end if
    else
      if (this%idiversions /= 0) then
        call store_error('REQUIRED DIVERSIONS BLOCK NOT FOUND.')
      end if
    end if
    !
    ! -- write summary of diversion error messages
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- return
    return
  end subroutine sfr_read_diversions


  subroutine sfr_rp(this)
! ******************************************************************************
! sfr_rp -- Read and Prepare
! Subroutine: (1) read itmp
!             (2) read new boundaries if itmp>0
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use TdisModule, only: kper, nper
    use InputOutputModule, only: urword
    use SimModule, only: ustop, store_error, count_errors
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    character(len=LINELENGTH) :: title
    character(len=LINELENGTH) :: line
    integer(I4B) :: ierr
    integer(I4B) :: n
    integer(I4B) :: ichkustrm
    logical :: isfound, endOfBlock
    ! -- formats
    character(len=*),parameter :: fmtblkerr = &
      "('Looking for BEGIN PERIOD iper.  Found ', a, ' instead.')"
    character(len=*),parameter :: fmtlsp = &
    &  "(1X,/1X,'REUSING ',A,'S FROM LAST STRESS PERIOD')"
    character(len=*), parameter :: fmtnbd = &
      "(1X,/1X,'The number of active ',A,'S (',I6, &
     &  ') is greater than maximum (',I6,')')"
! ------------------------------------------------------------------------------
    !
    ! -- initialize flags
    ichkustrm = 0
    if (kper == 1) then
      ichkustrm = 1
    end if
    !
    ! -- set nbound to maxbound
    this%nbound = this%maxbound
    !
    ! -- Set ionper to the stress period number for which a new block of data
    !    will be read.
    if (this%ionper < kper) then
      !
      ! -- get period block
      call this%parser%GetBlock('PERIOD', isfound, ierr, &
                                supportOpenClose=.true.)
      if(isfound) then
        !
        ! -- read ionper and check for increasing period numbers
        call this%read_check_ionper()
      else
        !
        ! -- PERIOD block not found
        if (ierr < 0) then
          ! -- End of file found; data applies for remainder of simulation.
          this%ionper = nper + 1
        else
          ! -- Found invalid block
          write(errmsg, fmtblkerr) adjustl(trim(line))
          call store_error(errmsg)
          call this%parser%StoreErrorUnit()
          call ustop()
        end if
      endif
    end if
    !
    ! -- Read data if ionper == kper
    if(this%ionper==kper) then
      !
      ! -- setup table for period data
      if (this%iprpak /= 0) then
        !
        ! -- reset the input table object
        title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
                trim(adjustl(this%packName)) //') DATA FOR PERIOD'
        write(title, '(a,1x,i6)') trim(adjustl(title)), kper
        call table_cr(this%inputtab, this%packName, title)
        call this%inputtab%table_df(1, 4, this%iout, finalize=.FALSE.)
        text = 'NUMBER'
        call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
        text = 'KEYWORD'
        call this%inputtab%initialize_column(text, 20, alignment=TABLEFT)
        do n = 1, 2
          write(text, '(a,1x,i6)') 'VALUE', n
          call this%inputtab%initialize_column(text, 15, alignment=TABCENTER)
        end do
      end if
      !
      ! -- read data
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        n = this%parser%GetInteger()
        if (n < 1 .or. n > this%maxbound) then
          write(errmsg,'(a,1x,a,1x,i0,a)') &
            'Reach number (RNO) must be greater than 0 and',                     &
            'less than or equal to', this%maxbound, '.'
          call store_error(errmsg)
          cycle
        end if
        !
        ! -- read data from the rest of the line
        call this%sfr_set_stressperiod(n, ichkustrm)
        !
        ! -- write line to table
        if (this%iprpak /= 0) then
          call this%parser%GetCurrentLine(line)
          call this%inputtab%line_to_columns(line)
        end if
      end do
      if (this%iprpak /= 0) then
        call this%inputtab%finalize_table()
      end if
    !
    ! -- Reuse data from last stress period
    else
      write(this%iout,fmtlsp) trim(this%filtyp)
    endif
    !
    ! -- check upstream fraction values
    if (ichkustrm /= 0) then
      call this%sfr_check_ustrf()
    end if
    !
    ! -- write summary of package block error messages
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- return
    return
  end subroutine sfr_rp

  subroutine sfr_ad(this)
! ******************************************************************************
! sfr_ad -- Add package connection to matrix
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use TimeSeriesManagerModule, only: var_timeseries
    ! -- dummy
    class(SfrType) :: this
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: iaux
! ------------------------------------------------------------------------------
    !
    ! -- Advance the time series manager
    call this%TsManager%ad()
    !
    ! -- check upstream fractions if time series are being used to
    !    define this variable
    if (var_timeseries(this%tsManager, this%packName, 'USTRF')) then
      call this%sfr_check_ustrf()
    end if
    !
    ! -- update auxiliary variables by copying from the derived-type time
    !    series variable into the bndpackage auxvar variable so that this
    !    information is properly written to the GWF budget file
    if (this%naux > 0) then
      do n = 1, this%maxbound
        do iaux = 1, this%naux
          if (this%noupdateauxvar(iaux) /= 0) cycle
          this%auxvar(iaux, n) = this%rauxvar(iaux, n)
        end do
      end do
    end if
    !
    ! -- reset upstream flow to zero and set specified stage
    do n = 1, this%maxbound
      this%usflow(n) = DZERO
      if (this%iboundpak(n) < 0) then
        this%stage(n) = this%sstage(n)
      end if
    end do
    !
    ! -- pakmvrobj ad
    if(this%imover == 1) then
      call this%pakmvrobj%ad()
    endif
    !
    ! -- For each observation, push simulated value and corresponding
    !    simulation time from "current" to "preceding" and reset
    !    "current" value.
    call this%obs%obs_ad()
    !
    ! -- return
    return
  end subroutine sfr_ad

  subroutine sfr_cf(this, reset_mover)
  ! ******************************************************************************
  ! sfr_cf -- Formulate the HCOF and RHS terms
  ! Subroutine: (1) skip in no wells
  !             (2) calculate hcof and rhs
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    ! -- dummy
    class(SfrType) :: this
    logical, intent(in), optional :: reset_mover
    ! -- local variables
    integer(I4B) :: n
    integer(I4B) :: igwfnode
    logical :: lrm
  ! ------------------------------------------------------------------------------
    !
    ! -- Return if no sfr reaches
    if(this%nbound == 0) return
    !
    ! -- find highest active cell
    do n = 1, this%nbound
      igwfnode = this%igwftopnode(n)
      if (igwfnode > 0) then
        if (this%ibound(igwfnode) == 0) then
          call this%dis%highest_active(igwfnode, this%ibound)
        end if
      end if
      this%igwfnode(n) = igwfnode
      this%nodelist(n) = igwfnode
    end do
    !
    ! -- pakmvrobj cf
    lrm = .true.
    if (present(reset_mover)) lrm = reset_mover
    if(this%imover == 1 .and. lrm) then
      call this%pakmvrobj%cf()
    endif
    !
    ! -- return
    return
  end subroutine sfr_cf

  subroutine sfr_fc(this, rhs, ia, idxglo, amatsln)
  ! **************************************************************************
  ! sfr_fc -- Copy rhs and hcof into solution rhs and amat
  ! **************************************************************************
  !
  !    SPECIFICATIONS:
  ! --------------------------------------------------------------------------
    ! -- dummy
    class(SfrType) :: this
    real(DP), dimension(:), intent(inout) :: rhs
    integer(I4B), dimension(:), intent(in) :: ia
    integer(I4B), dimension(:), intent(in) :: idxglo
    real(DP), dimension(:), intent(inout) :: amatsln
    ! -- local
    integer(I4B) :: i
    integer(I4B) :: n
    integer(I4B) :: ipos
    integer(I4B) :: node
    real(DP) :: s0
    real(DP) :: ds
    real(DP) :: dsmax
    real(DP) :: hgwf
    real(DP) :: v
    real(DP) :: hhcof
    real(DP) :: rrhs
! --------------------------------------------------------------------------
    !
    ! -- picard iterations for sfr to achieve good solution regardless
    !    of reach order
    sfrpicard: do i = 1, this%maxsfrpicard
      !
      ! -- initialize maximum stage change for iteration to zero
      dsmax = DZERO
      !
      ! -- pakmvrobj fc - reset qformvr to zero
      if(this%imover == 1) then
        call this%pakmvrobj%fc()
      endif
      !
      ! -- solve for each sfr reach
      reachsolve: do n = 1, this%nbound
        node = this%igwfnode(n)
        if (node > 0) then
          hgwf = this%xnew(node)
        else
          hgwf = DEP20
        end if
        !
        ! -- save previous stage and upstream flow
        if (i == 1) then
          this%stage0(n) = this%stage(n)
          this%usflow0(n) = this%usflow(n)
        end if
        !
        ! -- set initial stage to calculate stage change
        s0 = this%stage(n)
        !
        ! -- solve for flow in swr
        if (this%iboundpak(n) /= 0) then
          call this%sfr_solve(n, hgwf, hhcof, rrhs)
        else
          this%depth(n) = DZERO
          this%stage(n) = this%strtop(n)
          v = DZERO
          call this%sfr_update_flows(n, v, v)
          hhcof = DZERO
          rrhs = DZERO
        end if
        !
        ! -- set package hcof and rhs
        this%hcof(n) = hhcof
        this%rhs(n) = rrhs
        !
        ! -- calculate stage change
        ds = s0 - this%stage(n)
        !
        ! -- evaluate if stage change exceeds dsmax
        if (abs(ds) > abs(dsmax)) then
          dsmax = ds
        end if
        
      end do reachsolve
      !
      ! -- evaluate if the sfr picard iterations should be terminated
      if (abs(dsmax) <= this%dmaxchg) then
        exit sfrpicard
      end if
    
    end do sfrpicard
    !
    ! -- Copy package rhs and hcof into solution rhs and amat
    do n = 1, this%nbound
      node = this%nodelist(n)
      if (node < 1) cycle
      rhs(node) = rhs(node) + this%rhs(n)
      ipos = ia(node)
      amatsln(idxglo(ipos)) = amatsln(idxglo(ipos)) + this%hcof(n)
    end do
    !
    ! -- return
    return
  end subroutine sfr_fc

  subroutine sfr_fn(this, rhs, ia, idxglo, amatsln)
! **************************************************************************
! pak1fn -- Fill newton terms
! **************************************************************************
!
!    SPECIFICATIONS:
! --------------------------------------------------------------------------
    ! -- dummy
    class(SfrType) :: this
    real(DP), dimension(:), intent(inout) :: rhs
    integer(I4B), dimension(:), intent(in) :: ia
    integer(I4B), dimension(:), intent(in) :: idxglo
    real(DP), dimension(:), intent(inout) :: amatsln
    ! -- local
    integer(I4B) :: i, n
    integer(I4B) :: ipos
    real(DP) :: rterm, drterm
    real(DP) :: rhs1, hcof1, q1
    real(DP) :: q2
    real(DP) :: hgwf
! --------------------------------------------------------------------------
    !
    ! -- Copy package rhs and hcof into solution rhs and amat
    do i = 1, this%nbound
      ! -- skip inactive reaches
      if (this%iboundpak(i) < 1) cycle
      ! -- skip if reach is not connected to gwf
      n = this%nodelist(i)
      if (n < 1) cycle
      ipos = ia(n)
      rterm = this%hcof(i) * this%xnew(n)
      ! -- calculate perturbed head
      hgwf = this%xnew(n) + DEM4
      call this%sfr_solve(i, hgwf, hcof1, rhs1, update=.false.)
      q1 = rhs1 - hcof1 * hgwf
      ! -- calculate unperturbed head
      q2 = this%rhs(i) - this%hcof(i) * this%xnew(n)
      ! -- calculate derivative
      drterm = (q2 - q1) / DEM4
      ! -- add terms to convert conductance formulation into
      !    newton-raphson formulation
      amatsln(idxglo(ipos)) = amatsln(idxglo(ipos)) + drterm - this%hcof(i)
      rhs(n) = rhs(n) - rterm + drterm * this%xnew(n)
    end do
    !
    ! -- return
    return
  end subroutine sfr_fn

  subroutine sfr_cc(this, innertot, kiter, iend, icnvgmod, cpak, ipak, dpak)
! **************************************************************************
! sfr_cc -- Final convergence check for package
! **************************************************************************
!
!    SPECIFICATIONS:
! --------------------------------------------------------------------------
    use TdisModule, only: totim, kstp, kper, delt
    ! -- dummy
    class(SfrType), intent(inout) :: this
    integer(I4B), intent(in) :: innertot
    integer(I4B), intent(in) :: kiter
    integer(I4B), intent(in) :: iend
    integer(I4B), intent(in) :: icnvgmod
    character(len=LENPAKLOC), intent(inout) :: cpak
    integer(I4B), intent(inout) :: ipak
    real(DP), intent(inout) :: dpak
    ! -- local
    character(len=LENPAKLOC) :: cloc
    character(len=LINELENGTH) :: tag
    integer(I4B) :: icheck
    integer(I4B) :: ipakfail
    integer(I4B) :: locdhmax
    integer(I4B) :: locrmax
    integer(I4B) :: ntabrows
    integer(I4B) :: ntabcols
    integer(I4B) :: n
    real(DP) :: dh
    real(DP) :: r
    real(DP) :: dhmax
    real(DP) :: rmax
    ! format
! --------------------------------------------------------------------------
    !
    ! -- initialize local variables
    icheck = this%iconvchk 
    ipakfail = 0
    locdhmax = 0
    locrmax = 0
    dhmax = DZERO
    rmax = DZERO
    !
    ! -- if not saving package convergence data on check convergence if
    !    the model is considered converged
    if (this%ipakcsv == 0) then
      if (icnvgmod == 0) then
        icheck = 0
      end if
    !
    ! -- saving package convergence data
    else
      !
      ! -- header for package csv
      if (.not. associated(this%pakcsvtab)) then
        !
        ! -- determine the number of columns and rows
        ntabrows = 1
        ntabcols = 9
        !
        ! -- setup table
        call table_cr(this%pakcsvtab, this%packName, '')
        call this%pakcsvtab%table_df(ntabrows, ntabcols, this%ipakcsv,           &
                                     lineseparator=.FALSE., separator=',',       &
                                     finalize=.FALSE.)
        !
        ! -- add columns to package csv
        tag = 'total_inner_iterations'
        call this%pakcsvtab%initialize_column(tag, 10, alignment=TABLEFT)
        tag = 'totim'
        call this%pakcsvtab%initialize_column(tag, 10, alignment=TABLEFT)
        tag = 'kper'
        call this%pakcsvtab%initialize_column(tag, 10, alignment=TABLEFT)
        tag = 'kstp'
        call this%pakcsvtab%initialize_column(tag, 10, alignment=TABLEFT)
        tag = 'nouter'
        call this%pakcsvtab%initialize_column(tag, 10, alignment=TABLEFT)
        tag = 'dvmax'
        call this%pakcsvtab%initialize_column(tag, 15, alignment=TABLEFT)
        tag = 'dvmax_loc'
        call this%pakcsvtab%initialize_column(tag, 15, alignment=TABLEFT)
        tag = 'dinflowmax'
        call this%pakcsvtab%initialize_column(tag, 15, alignment=TABLEFT)
        tag = 'dinflowmax_loc'
        call this%pakcsvtab%initialize_column(tag, 15, alignment=TABLEFT)
      end if
    end if
    !
    ! -- perform package convergence check
    if (icheck /= 0) then
      final_check: do n = 1, this%maxbound
        if (this%iboundpak(n) == 0) cycle
        dh = this%stage0(n) - this%stage(n)
        r = this%usflow0(n) - this%usflow(n)
        !
        ! -- normalize flow difference and convert to a depth
        r = r * delt / this%surface_area(n)
        !
        ! -- evaluate magnitude of differences
        if (n == 1) then
          locdhmax = n
          dhmax = dh
          locrmax = n
          rmax = r
        else
          if (abs(dh) > abs(dhmax)) then
            locdhmax = n
            dhmax = dh
          end if
          if (abs(r) > abs(rmax)) then
            locrmax = n
            rmax = r
          end if
        end if
      end do final_check
      !
      ! -- set dpak and cpak
      if (ABS(dhmax) > abs(dpak)) then
        ipak = locdhmax
        dpak = dhmax
        write(cloc, "(a,'-',a)") trim(this%packName), 'stage'
        cpak = trim(cloc)
      end if
      if (ABS(rmax) > abs(dpak)) then
        ipak = locrmax
        dpak = rmax
        write(cloc, "(a,'-',a)") trim(this%packName), 'inflow'
        cpak = trim(cloc)
      end if
      !
      ! -- write convergence data to package csv
      if (this%ipakcsv /= 0) then
        !
        ! -- write the data
        call this%pakcsvtab%add_term(innertot)
        call this%pakcsvtab%add_term(totim)
        call this%pakcsvtab%add_term(kper)
        call this%pakcsvtab%add_term(kstp)
        call this%pakcsvtab%add_term(kiter)
        call this%pakcsvtab%add_term(dhmax)
        call this%pakcsvtab%add_term(locdhmax)
        call this%pakcsvtab%add_term(rmax)
        call this%pakcsvtab%add_term(locrmax)
        !
        ! -- finalize the package csv
        if (iend == 1) then
          call this%pakcsvtab%finalize_table()
        end if
      end if
    end if
    !
    ! -- return
    return
  end subroutine sfr_cc


  subroutine sfr_bd(this, x, idvfl, icbcfl, ibudfl, icbcun, iprobs,         &
                    isuppress_output, model_budget, imap, iadv)
! **************************************************************************
! bnd_bd -- Calculate Volumetric Budget
! Note that the compact budget will always be used.
! Subroutine: (1) Process each package entry
!             (2) Write output
! **************************************************************************
!
!    SPECIFICATIONS:
! --------------------------------------------------------------------------
    ! -- modules
    use TdisModule, only: kstp, kper, delt, pertim, totim
    use ConstantsModule, only: LENBOUNDNAME
    use InputOutputModule, only: ulasav, ubdsv06
    use BudgetModule, only: BudgetType
    ! -- dummy
    class(SfrType) :: this
    real(DP),dimension(:),intent(in) :: x
    integer(I4B), intent(in) :: idvfl
    integer(I4B), intent(in) :: icbcfl
    integer(I4B), intent(in) :: ibudfl
    integer(I4B), intent(in) :: icbcun
    integer(I4B), intent(in) :: iprobs
    integer(I4B), intent(in) :: isuppress_output
    type(BudgetType), intent(inout) :: model_budget
    integer(I4B), dimension(:), optional, intent(in) :: imap
    integer(I4B), optional, intent(in) :: iadv
    ! -- local
    integer(I4B) :: i
    integer(I4B) :: ibinun
    real(DP) :: qext
    ! -- for budget
    integer(I4B) :: n
    real(DP) :: d
    real(DP) :: v
    real(DP) :: qoutflow
    real(DP) :: qfrommvr
    real(DP) :: qtomvr
    ! -- for observations
    integer(I4B) :: iprobslocal
    ! -- formats
! --------------------------------------------------------------------------
    !
    ! -- Suppress saving of simulated values; they
    !    will be saved at end of this procedure.
    iprobslocal = 0
    !
    ! -- call base functionality in bnd_bd
    call this%BndType%bnd_bd(x, idvfl, icbcfl, ibudfl, icbcun, iprobslocal,    &
                             isuppress_output, model_budget, iadv=1)
    !
    ! -- Calculate qextoutflow and qoutflow for subsequent budgets
    do n = 1, this%maxbound
      !
      ! -- mover
      qfrommvr = DZERO
      qtomvr = DZERO
      if (this%imover == 1) then
        qfrommvr = this%pakmvrobj%get_qfrommvr(n)
        qtomvr = this%pakmvrobj%get_qtomvr(n)
        if (qtomvr > DZERO) then
          qtomvr = -qtomvr
        end if
      endif
      !
      ! -- external downstream stream flow
      qext = this%dsflow(n)
      qoutflow = DZERO
      if (qext > DZERO) then
        qext = -qext
      end if
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        if (this%idir(i) > 0) cycle
        qext = DZERO
        exit
      end do
      !
      ! -- adjust external downstream stream flow using qtomvr
      if (qext < DZERO) then
        if (qtomvr < DZERO) then
          qext = qext - qtomvr
        end if
      else
        qoutflow = this%dsflow(n)
        if (qoutflow > DZERO) then
          qoutflow = -qoutflow
        end if
      end if
      !
      ! -- set qextoutflow and qoutflow for cell by cell budget
      !    output and observations
      this%qextoutflow(n) = qext
      this%qoutflow(n) = qoutflow
      !
    end do
    !
    ! -- For continuous observations, save simulated values.
    if (this%obs%npakobs > 0 .and. iprobs > 0) then
      call this%sfr_bd_obs()
    end if
    !
    ! -- set unit number for binary dependent variable output
    ibinun = 0
    if(this%istageout /= 0) then
      ibinun = this%istageout
    end if
    if(idvfl == 0) ibinun = 0
    if (isuppress_output /= 0) ibinun = 0
    !
    ! -- write sfr binary output
    if (ibinun > 0) then
      do n = 1, this%maxbound
        d = this%depth(n)
        v = this%stage(n)
        if (this%iboundpak(n) == 0) then
          v = DHNOFLO
        else if (d == DZERO) then
          v = DHDRY
        end if
        this%dbuff(n) = v
      end do
      call ulasav(this%dbuff, '           STAGE', kstp, kper, pertim, totim,   &
                  this%maxbound, 1, 1, ibinun)
    end if
    !
    ! -- fill the budget object
    call this%sfr_fill_budobj()
    !
    ! -- write the flows from the budobj
    ibinun = 0
    if(this%ibudgetout /= 0) then
      ibinun = this%ibudgetout
    end if
    if(icbcfl == 0) ibinun = 0
    if (isuppress_output /= 0) ibinun = 0
    if (ibinun > 0) then
      call this%budobj%save_flows(this%dis, ibinun, kstp, kper, delt, &
                                  pertim, totim, this%iout)
    end if
    !
    !
    ! -- return
    return
  end subroutine sfr_bd

  subroutine sfr_ot(this, kstp, kper, iout, ihedfl, ibudfl)
! ******************************************************************************
! sfr_ot -- Output package budget
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(SfrType) :: this
    integer(I4B),intent(in) :: kstp
    integer(I4B),intent(in) :: kper
    integer(I4B),intent(in) :: iout
    integer(I4B),intent(in) :: ihedfl
    integer(I4B),intent(in) :: ibudfl
    ! -- locals
    character (len=20) :: cellid
    character (len=20), dimension(:), allocatable :: cellidstr
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: hgwf
    real(DP) :: sbot
    real(DP) :: depth, stage
    real(DP) :: w, cond, grad
    ! -- format
! ------------------------------------------------------------------------------
     !
     ! -- write sfr stage and depth table
     if (ihedfl /= 0 .and. this%iprhed /= 0) then
      !
      ! -- set table kstp and kper
      call this%stagetab%set_kstpkper(kstp, kper)
      !
      ! -- fill stage data
      do n = 1, this%maxbound
        node = this%igwfnode(n)
        if (node > 0) then
          call this%dis%noder_to_string(node, cellid)
          hgwf = this%xnew(node)
        else
          cellid = 'NONE'
        end if
        if(this%inamedbound==1) then
          call this%stagetab%add_term(this%boundname(n))
        end if
        call this%stagetab%add_term(n)
        call this%stagetab%add_term(cellid)
        depth = this%depth(n)
        stage = this%stage(n)
        w = this%top_width_wet(n, depth)
        call this%stagetab%add_term(stage)
        call this%stagetab%add_term(depth)
        call this%stagetab%add_term(w)
        call this%sfr_calc_cond(n, cond)
        if (node > 0) then
          sbot = this%strtop(n) - this%bthick(n)
          if (hgwf < sbot) then
            grad = stage - sbot
          else
            grad = stage - hgwf
          end if
          grad = grad / this%bthick(n)
          call this%stagetab%add_term(hgwf)
          call this%stagetab%add_term(cond)
          call this%stagetab%add_term(grad)
        else
          call this%stagetab%add_term('--')
          call this%stagetab%add_term('--')
          call this%stagetab%add_term('--')
        end if
      end do
     end if
    !
    ! -- Output sfr flow table
    if (ibudfl /= 0 .and. this%iprflow /= 0) then
      !
      ! -- If there are any 'none' gwf connections then need to calculate
      !    a vector of cellids and pass that in to the budget flow table because
      !    the table assumes that there are maxbound gwf entries, which is not
      !    the case if any 'none's are specified.
      if (this%ianynone > 0) then
        allocate(cellidstr(this%maxbound))
        do n = 1, this%maxbound
          node = this%igwfnode(n)
          if (node > 0) then
            call this%dis%noder_to_string(node, cellidstr(n))
          else
            cellidstr(n) = 'NONE'
          end if
        end do
        call this%budobj%write_flowtable(this%dis, kstp, kper, cellidstr)
        deallocate(cellidstr)
      else
        call this%budobj%write_flowtable(this%dis, kstp, kper)
      end if
    end if
    !
    ! -- Output sfr budget
    call this%budobj%write_budtable(kstp, kper, iout)
    !
    ! -- return
    return
  end subroutine sfr_ot

  subroutine sfr_da(this)
! ******************************************************************************
! sfr_da -- deallocate
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_deallocate
    ! -- dummy
    class(SfrType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- arrays
    call mem_deallocate(this%qoutflow)
    call mem_deallocate(this%qextoutflow)
    deallocate(this%csfrbudget)
    call mem_deallocate(this%sfrname, 'SFRNAME', this%memoryPath)
    call mem_deallocate(this%dbuff)
    deallocate(this%cauxcbc)
    call mem_deallocate(this%qauxcbc)
    call mem_deallocate(this%iboundpak)
    call mem_deallocate(this%igwfnode)
    call mem_deallocate(this%igwftopnode)
    call mem_deallocate(this%length)
    call mem_deallocate(this%width)
    call mem_deallocate(this%strtop)
    call mem_deallocate(this%bthick)
    call mem_deallocate(this%hk)
    call mem_deallocate(this%slope)
    call mem_deallocate(this%nconnreach)
    call mem_deallocate(this%ustrf)
    call mem_deallocate(this%ftotnd)
    call mem_deallocate(this%usflow)
    call mem_deallocate(this%dsflow)
    call mem_deallocate(this%depth)
    call mem_deallocate(this%stage)
    call mem_deallocate(this%gwflow)
    call mem_deallocate(this%simevap)
    call mem_deallocate(this%simrunoff)
    call mem_deallocate(this%stage0)
    call mem_deallocate(this%usflow0)
    call mem_deallocate(this%denseterms)
    !
    ! -- connection data
    call mem_deallocate(this%ia)
    call mem_deallocate(this%ja)
    call mem_deallocate(this%idir)
    call mem_deallocate(this%idiv)
    call mem_deallocate(this%qconn)
    !
    ! -- boundary data
    call mem_deallocate(this%rough)
    call mem_deallocate(this%rain)
    call mem_deallocate(this%evap)
    call mem_deallocate(this%inflow)
    call mem_deallocate(this%runoff)
    call mem_deallocate(this%sstage)
    !
    ! -- aux variables
    call mem_deallocate(this%rauxvar)
    !
    ! -- diversion variables
    call mem_deallocate(this%iadiv)
    call mem_deallocate(this%divreach)
    if (associated(this%divcprior)) then
      deallocate(this%divcprior)
    end if
    call mem_deallocate(this%divflow)
    call mem_deallocate(this%divq)
    call mem_deallocate(this%ndiv)
    !
    ! -- budobj
    call this%budobj%budgetobject_da()
    deallocate(this%budobj)
    nullify(this%budobj)
    !
    ! -- stage table
    if (this%iprhed > 0) then
      call this%stagetab%table_da()
      deallocate(this%stagetab)
      nullify(this%stagetab)
    end if
    !
    ! -- package csv table
    if (this%ipakcsv > 0) then
      call this%pakcsvtab%table_da()
      deallocate(this%pakcsvtab)
      nullify(this%pakcsvtab)
    end if
    !
    ! -- scalars
    call mem_deallocate(this%iprhed)
    call mem_deallocate(this%istageout)
    call mem_deallocate(this%ibudgetout)
    call mem_deallocate(this%ipakcsv)
    call mem_deallocate(this%idiversions)
    call mem_deallocate(this%maxsfrpicard)
    call mem_deallocate(this%maxsfrit)
    call mem_deallocate(this%bditems)
    call mem_deallocate(this%cbcauxitems)
    call mem_deallocate(this%unitconv)
    call mem_deallocate(this%dmaxchg)
    call mem_deallocate(this%deps)
    call mem_deallocate(this%nconn)
    call mem_deallocate(this%icheck)
    call mem_deallocate(this%iconvchk)
    call mem_deallocate(this%idense)
    call mem_deallocate(this%ianynone)
    nullify(this%gwfiss)
    !
    ! -- call BndType deallocate
    call this%BndType%bnd_da()
    !
    ! -- return
  end subroutine sfr_da

  subroutine define_listlabel(this)
! ******************************************************************************
! define_listlabel -- Define the list heading that is written to iout when
!   PRINT_INPUT option is used.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    class(SfrType), intent(inout) :: this
! ------------------------------------------------------------------------------
    !
    ! -- create the header list label
    this%listlabel = trim(this%filtyp) // ' NO.'
    if(this%dis%ndim == 3) then
      write(this%listlabel, '(a, a7)') trim(this%listlabel), 'LAYER'
      write(this%listlabel, '(a, a7)') trim(this%listlabel), 'ROW'
      write(this%listlabel, '(a, a7)') trim(this%listlabel), 'COL'
    elseif(this%dis%ndim == 2) then
      write(this%listlabel, '(a, a7)') trim(this%listlabel), 'LAYER'
      write(this%listlabel, '(a, a7)') trim(this%listlabel), 'CELL2D'
    else
      write(this%listlabel, '(a, a7)') trim(this%listlabel), 'NODE'
    endif
    write(this%listlabel, '(a, a16)') trim(this%listlabel), 'STRESS RATE'
    if(this%inamedbound == 1) then
      write(this%listlabel, '(a, a16)') trim(this%listlabel), 'BOUNDARY NAME'
    endif
    !
    ! -- return
    return
  end subroutine define_listlabel


  subroutine sfr_set_pointers(this, neq, ibound, xnew, xold, flowja)
! ******************************************************************************
! set_pointers -- Set pointers to model arrays and variables so that a package
!                 has access to these things.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    class(SfrType) :: this
    integer(I4B), pointer :: neq
    integer(I4B), dimension(:), pointer, contiguous :: ibound
    real(DP), dimension(:), pointer, contiguous :: xnew
    real(DP), dimension(:), pointer, contiguous :: xold
    real(DP), dimension(:), pointer, contiguous :: flowja
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- call base BndType set_pointers
    call this%BndType%set_pointers(neq, ibound, xnew, xold, flowja)
    !
    ! -- return
  end subroutine sfr_set_pointers

  !
  ! -- Procedures related to observations (type-bound)
  logical function sfr_obs_supported(this)
  ! ******************************************************************************
  ! sfr_obs_supported
  !   -- Return true because sfr package supports observations.
  !   -- Overrides BndType%bnd_obs_supported()
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    class(SfrType) :: this
  ! ------------------------------------------------------------------------------
    sfr_obs_supported = .true.
    return
  end function sfr_obs_supported


  subroutine sfr_df_obs(this)
  ! ******************************************************************************
  ! sfr_df_obs (implements bnd_df_obs)
  !   -- Store observation type supported by sfr package.
  !   -- Overrides BndType%bnd_df_obs
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
    ! -- dummy
    class(SfrType) :: this
    ! -- local
    integer(I4B) :: indx
  ! ------------------------------------------------------------------------------
    !
    ! -- Store obs type and assign procedure pointer
    !    for stage observation type.
    call this%obs%StoreObsType('stage', .false., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for inflow observation type.
    call this%obs%StoreObsType('inflow', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for inflow observation type.
    call this%obs%StoreObsType('ext-inflow', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for rainfall observation type.
    call this%obs%StoreObsType('rainfall', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for runoff observation type.
    call this%obs%StoreObsType('runoff', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for evaporation observation type.
    call this%obs%StoreObsType('evaporation', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for outflow observation type.
    call this%obs%StoreObsType('outflow', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for ext-outflow observation type.
    call this%obs%StoreObsType('ext-outflow', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for to-mvr observation type.
    call this%obs%StoreObsType('to-mvr', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for sfr-frommvr observation type.
    call this%obs%StoreObsType('from-mvr', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for sfr observation type.
    call this%obs%StoreObsType('sfr', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for upstream flow observation type.
    call this%obs%StoreObsType('upstream-flow', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for downstream flow observation type.
    call this%obs%StoreObsType('downstream-flow', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => sfr_process_obsID
    !
    return
  end subroutine sfr_df_obs


  subroutine sfr_bd_obs(this)
    ! **************************************************************************
    ! sfr_bd_obs
    !   -- Calculate observations this time step and call
    !      ObsType%SaveOneSimval for each SfrType observation.
    ! **************************************************************************
    !
    !    SPECIFICATIONS:
    ! --------------------------------------------------------------------------
    ! -- dummy
    class(SfrType), intent(inout) :: this
    ! -- local
    integer(I4B) :: i
    integer(I4B) :: j
    integer(I4B) :: n
    real(DP) :: v
    character(len=100) :: msg
    type(ObserveType), pointer :: obsrv => null()
    !---------------------------------------------------------------------------
    !
    ! Write simulated values for all sfr observations
    if (this%obs%npakobs > 0) then
      call this%obs%obs_bd_clear()
      do i=1 ,this%obs%npakobs
        obsrv => this%obs%pakobs(i)%obsrv
        do j = 1, obsrv%indxbnds_count
          n = obsrv%indxbnds(j)
          v = DZERO
          select case (obsrv%ObsTypeId)
            case ('STAGE')
              v = this%stage(n)
            case ('TO-MVR')
              v = DNODATA
              if (this%imover == 1) then
                v = this%pakmvrobj%get_qtomvr(n)
                if (v > DZERO) then
                  v = -v
                end if
              end if
            case ('FROM-MVR')
              v = DNODATA
              if (this%imover == 1) then
                v = this%pakmvrobj%get_qfrommvr(n)
              end if
            case ('EXT-INFLOW')
              v = this%inflow(n)
            case ('INFLOW')
              v = this%usflow(n)
            case ('OUTFLOW')
              v = this%qoutflow(n)
            case ('EXT-OUTFLOW')
              v = this%qextoutflow(n)
            case ('RAINFALL')
              v = this%rain(n)
            case ('RUNOFF')
              v = this%simrunoff(n)
            case ('EVAPORATION')
              v = this%simevap(n)
            case ('SFR')
              v = this%gwflow(n)
            case ('UPSTREAM-FLOW')
              v = this%usflow(n)
              if (this%imover == 1) then
                v = v + this%pakmvrobj%get_qfrommvr(n)
              end if
            case ('DOWNSTREAM-FLOW')
              v = this%dsflow(n)
              if (v > DZERO) then
                v = -v
              end if
            case default
              msg = 'Unrecognized observation type: ' // trim(obsrv%ObsTypeId)
              call store_error(msg)
          end select
          call this%obs%SaveOneSimval(obsrv, v)
        end do
      end do
      !
      ! -- write summary of package error messages
      if (count_errors() > 0) then
        call this%parser%StoreErrorUnit()
        call ustop()
      end if
    end if
    !
    return
  end subroutine sfr_bd_obs


  subroutine sfr_rp_obs(this)
    use TdisModule, only: kper
    ! -- dummy
    class(SfrType), intent(inout) :: this
    ! -- local
    integer(I4B) :: i
    integer(I4B) :: j
    integer(I4B) :: nn1
    character(len=LENBOUNDNAME) :: bname
    logical :: jfound
    class(ObserveType),   pointer :: obsrv => null()
    ! --------------------------------------------------------------------------
    ! -- formats
10  format('Boundary "',a,'" for observation "',a,                               &
           '" is invalid in package "',a,'"')
30  format('Boundary name not provided for observation "',a,                     &
           '" in package "',a,'"')
    !
    ! -- process each package observation
    !    only done the first stress period since boundaries are fixed
    !    for the simulation
    if (kper == 1) then
      do i = 1, this%obs%npakobs
        obsrv => this%obs%pakobs(i)%obsrv
        !
        ! -- get node number 1
        nn1 = obsrv%NodeNumber
        if (nn1 == NAMEDBOUNDFLAG) then
          bname = obsrv%FeatureName
          if (bname /= '') then
            ! -- Observation location(s) is(are) based on a boundary name.
            !    Iterate through all boundaries to identify and store
            !    corresponding index(indices) in bound array.
            jfound = .false.
            do j = 1, this%maxbound
              if (this%boundname(j) == bname) then
                jfound = .true.
                call obsrv%AddObsIndex(j)
              endif
            enddo
            if (.not. jfound) then
              write(errmsg,10) trim(bname), trim(obsrv%name), trim(this%packName)
              call store_error(errmsg)
            endif
          else
            write(errmsg,30) trim(obsrv%name), trim(this%packName)
            call store_error(errmsg)
          endif
        else if (nn1 < 1 .or. nn1 > this%maxbound) then
          write(errmsg, '(a,1x,a,1x,i0,1x,a,1x,i0,a)')                             &
            trim(adjustl(obsrv%ObsTypeId)),                                        &
            'reach must be greater than 0 and less than or equal to',              &
            this%maxbound, '(specified value is ', nn1, ')'
          call store_error(errmsg)
        else
          if (obsrv%indxbnds_count == 0) then
            call obsrv%AddObsIndex(nn1)
          else
            errmsg = 'Programming error in sfr_rp_obs'
            call store_error(errmsg)
          end if
        end if
        !
        ! -- catch non-cumulative observation assigned to observation defined
        !    by a boundname that is assigned to more than one element
        if (obsrv%ObsTypeId == 'STAGE') then
          nn1 = obsrv%NodeNumber
          if (nn1 == NAMEDBOUNDFLAG) then
            if (obsrv%indxbnds_count > 1) then
              write(errmsg, '(a,3(1x,a))')                                         &
                trim(adjustl(obsrv%ObsTypeId)),                                    &
                'for observation', trim(adjustl(obsrv%Name)),                      &
                ' must be assigned to a reach with a unique boundname.'
              call store_error(errmsg)
            end if
          end if
        end if
        !
        ! -- check that node number 1 is valid; call store_error if not
        do j = 1, obsrv%indxbnds_count
          nn1 = obsrv%indxbnds(j)
          if (nn1 < 1 .or. nn1 > this%maxbound) then
            write(errmsg, '(a,1x,a,1x,i0,1x,a,1x,i0,a)')                           &
              trim(adjustl(obsrv%ObsTypeId)),                                      &
              'reach must be greater than 0 and less than or equal to',            &
              this%maxbound, '(specified value is ', nn1, ')'
            call store_error(errmsg)
          end if
        end do
      end do
      !
      ! -- evaluate if there are any observation errors
      if (count_errors() > 0) then
        call this%parser%StoreErrorUnit()
        call ustop()
      end if
    end if
    !
    return
  end subroutine sfr_rp_obs


  !
  ! -- Procedures related to observations (NOT type-bound)
  subroutine sfr_process_obsID(obsrv, dis, inunitobs, iout)
    ! -- This procedure is pointed to by ObsDataType%ProcesssIdPtr. It processes
    !    the ID string of an observation definition for sfr-package observations.
    ! -- dummy
    type(ObserveType),      intent(inout) :: obsrv
    class(DisBaseType), intent(in)    :: dis
    integer(I4B),            intent(in)    :: inunitobs
    integer(I4B),            intent(in)    :: iout
    ! -- local
    integer(I4B) :: nn1
    integer(I4B) :: icol, istart, istop
    character(len=LINELENGTH) :: strng
    character(len=LENBOUNDNAME) :: bndname
    ! formats
    !
    strng = obsrv%IDstring
    !
    ! -- Extract reach number from strng and store it.
    !    If 1st item is not an integer(I4B), it should be a
    !    boundary name--deal with it.
    icol = 1
    !
    ! -- get reach number or boundary name
    call extract_idnum_or_bndname(strng, icol, istart, istop, nn1, bndname)
    if (nn1 == NAMEDBOUNDFLAG) then
      obsrv%FeatureName = bndname
    endif
    !
    ! -- store reach number (NodeNumber)
    obsrv%NodeNumber = nn1
    !
    return
  end subroutine sfr_process_obsID

  !
  ! -- private sfr methods
  !


  subroutine sfr_set_stressperiod(this, n, ichkustrm)
! ******************************************************************************
! sfr_set_stressperiod -- Set a stress period attribute for sfr reach n
!                         using keywords.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use TimeSeriesManagerModule, only: read_value_or_time_series_adv
    use ConstantsModule, only: DZERO, DONE
    use SimModule, only: ustop, store_error
    ! -- dummy
    class(SfrType),intent(inout) :: this
    integer(I4B), intent(in) :: n
    integer(I4B), intent(inout) :: ichkustrm
    ! -- local
    character(len=10) :: cnum
    character(len=LINELENGTH) :: text
    character(len=LINELENGTH) :: caux
    character(len=LINELENGTH) :: keyword
    integer(I4B) :: ival
    integer(I4B) :: ii
    integer(I4B) :: jj
    integer(I4B) :: idiv
    character (len=10) :: cp
    real(DP) :: divq
    real(DP), pointer :: bndElem => null()
    ! -- formats
! ------------------------------------------------------------------------------
    !
    ! -- read line
    call this%parser%GetStringCaps(keyword)
    select case (keyword)
      case ('STATUS')
        ichkustrm = 1
        call this%parser%GetStringCaps(text)
        if (text == 'INACTIVE') then
          this%iboundpak(n) = 0
        else if (text == 'ACTIVE') then
          this%iboundpak(n) = 1
        else if (text == 'SIMPLE') then
          this%iboundpak(n) = -1
        else
          write(errmsg,'(2a)') &
            'Unknown ' // trim(this%text) // ' sfr status keyword: ', trim(text)
          call store_error(errmsg)
        end if
      case ('MANNING')
        call this%parser%GetString(text)
        jj = 1  ! For 'MANNING'
        bndElem => this%rough(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'MANNING')
      case ('STAGE')
        call this%parser%GetString(text)
        jj = 1  ! For 'STAGE'
        bndElem => this%sstage(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'STAGE')
      case ('RAINFALL')
        call this%parser%GetString(text)
        jj = 1  ! For 'RAIN'
        bndElem => this%rain(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'RAIN')
      case ('EVAPORATION')
        call this%parser%GetString(text)
        jj = 1  ! For 'EVAP'
        bndElem => this%evap(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'MANNING')
      case ('RUNOFF')
        call this%parser%GetString(text)
        jj = 1  ! For 'RUNOFF'
        bndElem => this%runoff(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'RUNOFF')
     case ('INFLOW')
        call this%parser%GetString(text)
        jj = 1  ! For 'INFLOW'
        bndElem => this%inflow(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'INFLOW')
      case ('DIVERSION')
        !
        ! -- make sure reach has at least one diversion
        if (this%ndiv(n) < 1) then
          write(cnum, '(i0)') n
          errmsg = 'diversions cannot be specified for reach ' // trim(cnum)
          call store_error(errmsg)
        end if
        !
        ! -- read diversion number
        ival = this%parser%GetInteger()
        if (ival < 1 .or. ival > this%ndiv(n)) then
          write(cnum, '(i0)') n
          errmsg = 'Reach  ' // trim(cnum)
          write(cnum, '(i0)') this%ndiv(n)
          errmsg = trim(errmsg) // ' diversion number should be between 1 ' //   &
                   'and ' // trim(cnum) // '.'
          call store_error(errmsg)
        end if
        idiv = ival
        !
        ! -- read value
        call this%parser%GetString(text)
        ii = this%iadiv(n) + idiv - 1
        jj = 1  ! For 'DIVERSION'
        bndElem => this%divflow(ii)
        call read_value_or_time_series_adv(text, ii, jj, bndElem, this%packName,     &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'DIVFLOW')
        !
        ! -- if diversion cprior is 'fraction', ensure that 0.0 <= fraction <= 1.0
        cp = this%divcprior(ii)
        divq = this%divflow(ii) 
        if (cp == 'FRACTION' .and. (divq < DZERO .or. divq > DONE)) then
          write(errmsg,'(a,1x,i0,a)')                                            &
                'cprior is type FRACTION for diversion no.', ii,                 &
                ', but divflow not within the range 0.0 to 1.0'
          call store_error(errmsg)
        endif
      case ('UPSTREAM_FRACTION')
        ichkustrm = 1
        call this%parser%GetString(text)
        jj = 1  ! For 'USTRF'
        bndElem => this%ustrf(n)
        call read_value_or_time_series_adv(text, n, jj, bndElem, this%packName,      &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'USTRF')
      case ('AUXILIARY')
        call this%parser%GetStringCaps(caux)
        do jj = 1, this%naux
          if (trim(adjustl(caux)) /= trim(adjustl(this%auxname(jj)))) cycle
          call this%parser%GetString(text)
          ii = n
          bndElem => this%rauxvar(jj, ii)
          call read_value_or_time_series_adv(text, ii, jj, bndElem, this%packName,   &
                                             'AUX', this%tsManager, this%iprpak, &
                                             this%auxname(jj))
          exit
        end do

      case default
        write(errmsg,'(a,a)') &
          'Unknown ' // trim(this%text) // ' sfr data keyword: ',                &
          trim(keyword) // '.'
        call store_error(errmsg)
      end select
    !
    ! -- return
    return
  end subroutine sfr_set_stressperiod


  subroutine sfr_solve(this, n, h, hcof, rhs, update)
  ! ******************************************************************************
  ! sfr_solve -- Solve continuity equation
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(in) :: h
      real(DP), intent(inout) :: hcof
      real(DP), intent(inout) :: rhs
      logical, intent(in), optional :: update
      ! -- local
      logical :: lupdate
      integer(I4B) :: i, ii
      integer(I4B) :: n2
      integer(I4B) :: isolve
      integer(I4B) :: iic, iic2, iic3, iic4
      integer(I4B) :: ibflg
      real(DP) :: hgwf
      real(DP) :: qu, qi, qr, qe, qro, qmp, qsrc
      real(DP) :: qfrommvr
      real(DP) :: qgwf
      real(DP) :: qmpsrc
      real(DP) :: qc
      real(DP) :: qt
      real(DP) :: tp
      real(DP) :: bt
      real(DP) :: hsfr
      real(DP) :: cstr
      real(DP) :: qd
      real(DP) :: en1, en2
      real(DP) :: qen1
      real(DP) :: f1, f2
      real(DP) :: qgwf1, qgwf2, qgwfp, qgwfold
      real(DP) :: fhstr1, fhstr2
      real(DP) :: d1, d2, dpp, dx
      real(DP) :: q1, q2
      real(DP) :: derv
      real(DP) :: dlh, dlhold
      real(DP) :: fp
      real(DP) :: err, errold
      real(DP) :: sumleak, sumrch
      real(DP) :: gwfhcof, gwfrhs
  ! ------------------------------------------------------------------------------
    !
    ! -- Process optional dummy variables
    if (present(update)) then
      lupdate = update
    else
      lupdate = .true.
    end if
    !
    ! -- calculate hgwf
    hgwf = h
    !
    !
    hcof = DZERO
    rhs = DZERO
    !
    ! -- initialize d1, d2, q1, q2, qsrc, and qgwf
    d1 = DZERO
    d2 = DZERO
    q1 = DZERO
    q2 = DZERO
    qsrc = DZERO
    qgwf = DZERO
    qgwfold = DZERO
    !
    ! -- calculate initial depth assuming a wide cross-section and ignore
    !    groundwater leakage
    ! -- calculate upstream flow
    qu = DZERO
    do i = this%ia(n) + 1, this%ia(n+1) - 1
      if (this%idir(i) < 0) cycle
      n2 = this%ja(i)
      do ii = this%ia(n2) + 1, this%ia(n2+1) - 1
        if (this%idir(ii) > 0) cycle
        if (this%ja(ii) /= n) cycle
        qu = qu + this%qconn(ii)
      end do
    end do
    this%usflow(n) = qu
    ! -- calculate remaining terms
    qi = this%inflow(n)
    qr = this%rain(n) * this%width(n) * this%length(n)
    qe = this%evap(n) * this%width(n) * this%length(n)
    qro = this%runoff(n)
    !
    ! -- Water mover term; assume that it goes in at the upstream end of the reach
    qfrommvr = DZERO
    if(this%imover == 1) then
      qfrommvr = this%pakmvrobj%get_qfrommvr(n)
    endif
    !
    ! -- calculate sum of sources to the reach excluding groundwater leakage
    qc = qu + qi + qr - qe + qro + qfrommvr
    !
    ! -- adjust runoff or evaporation if sum of sources is negative
    if (qc < DZERO) then
      !
      ! -- calculate sources without et
      qt = qu + qi + qr + qro + qfrommvr
      !
      ! -- runoff exceeds sources of water for reach
      if (qt < DZERO) then
        qro = -(qu + qi + qr + qfrommvr)
        qe = DZERO
      !
      ! -- evaporation exceeds sources of water for reach
      else
        qe = qu + qi + qr + qro + qfrommvr
      end if
      qc = qu + qi + qr - qe + qro + qfrommvr
    end if
    !
    ! -- set simulated evaporation and runoff
    this%simevap(n) = qe
    this%simrunoff(n) = qro
    !
    ! -- calculate flow at the middle of the reach and excluding groundwater leakage
    qmp = qu + qi + qfrommvr + DHALF * (qr - qe + qro)
    qmpsrc = qmp
    !
    ! -- calculate stream depth at the midpoint
    if (this%iboundpak(n) > 0) then
      call this%sfr_rectch_depth(n, qmp, d1)
    else
      this%stage(n) = this%sstage(n)
      d1 = max(DZERO, this%stage(n) - this%strtop(n))
    end if
    !
    ! -- calculate sources/sinks for reach excluding groundwater leakage
    call this%sfr_calc_qsource(n, d1, qsrc)
    !
    ! -- calculate initial reach stage, downstream flow, and groundwater leakage
    tp = this%strtop(n)
    bt = tp - this%bthick(n)
    hsfr = d1 + tp
    qd = MAX(qsrc, DZERO)
    qgwf = DZERO
    !
    ! -- calculate reach conductance for a unit depth of water
    !    if equal to zero will skip iterations
    call this%sfr_calc_cond(n, cstr)
    !
    ! -- set flag to skip iterations
    isolve = 1
    if (hsfr <= tp .and. hgwf <= tp) isolve = 0
    if (hgwf <= tp .and. qc < DEM30) isolve = 0
    if (cstr < DEM30) isolve = 0
    if (this%iboundpak(n) < 0) isolve = 0
    !
    ! -- iterate to achieve solution
    itersol: if (isolve /= 0) then
      !
      ! -- estimate initial end points
      en1 = DZERO
      if (d1 > DEM30) then
        if ((tp - hgwf) > DEM30) then
          en2 = DP9 * d1
        else
          en2 = D1P1 * d1 - (tp - hgwf)
        end if
      else if ((tp - hgwf) > DEM30) then
        en2 = DONE
      else
        en2 = DP99 * (hgwf - tp)
      end if
      !
      ! -- estimate flow at end points
      ! -- end point 1
      if (hgwf > tp) then
        call this%sfr_calc_qgwf(n, DZERO, hgwf, qgwf1)
        qgwf1 = -qgwf1
        qen1 = qmp - DHALF * qgwf1
      else
        qgwf1 = DZERO
        qen1 = qmpsrc
      end if
      if (hgwf > bt) then
        call this%sfr_calc_qgwf(n, en2, hgwf, qgwf2)
        qgwf2 = -qgwf2
      else
        call this%sfr_calc_qgwf(n, en2, bt, qgwf2)
        qgwf2 = -qgwf2
      end if
      if (qgwf2 > qsrc) qgwf2 = qsrc
      ! -- calculate two depths
      call this%sfr_rectch_depth(n, (qmpsrc-DHALF*qgwf1), d1)
      call this%sfr_rectch_depth(n, (qmpsrc-DHALF*qgwf2), d2)
      ! -- determine roots
      if (d1 > DEM30) then
        f1 = en1 - d1
      else
        en1 = DZERO
        f1 = en1 - DZERO
      end if
      if (d2 > DEM30) then
        f2 = en2 - d2
        if (f2 < DEM30) en2 = d2
      else
        d2 = DZERO
        f2 = en2 - DZERO
      end if
      !
      ! -- iterate to find a solution
      dpp = DHALF * (en1 + en2)
      dx = dpp
      iic = 0
      iic2 = 0
      iic3 = 0
      fhstr1 = DZERO
      fhstr2 = DZERO
      qgwfp = DZERO
      dlhold = DZERO
      do i = 1, this%maxsfrit
        ibflg = 0
        d1 = dpp
        d2 = d1 + DTWO * this%deps
        ! -- calculate q at midpoint at both end points
        call this%sfr_calc_qman(n, d1, q1)
        call this%sfr_calc_qman(n, d2, q2)
        ! -- calculate groundwater leakage at both end points
        call this%sfr_calc_qgwf(n, d1, hgwf, qgwf1)
        qgwf1 = -qgwf1
        call this%sfr_calc_qgwf(n, d2, hgwf, qgwf2)
        qgwf2 = -qgwf2
        !
        if (qgwf1 >= qsrc) then
          en2 = dpp
          dpp = DHALF * (en1 + en2)
          call this%sfr_calc_qgwf(n, dpp, hgwf, qgwfp)
          qgwfp = -qgwfp
          if (qgwfp > qsrc) qgwfp = qsrc
          call this%sfr_rectch_depth(n, (qmpsrc-DHALF*qgwfp), dx)
          ibflg = 1
        else
          fhstr1 = (qmpsrc-DHALF*qgwf1) - q1
          fhstr2 = (qmpsrc-DHALF*qgwf2) - q2
        end if
        !
        if (ibflg == 0) then
          derv = DZERO
          if (abs(d1-d2) > DZERO) then
            derv = (fhstr1-fhstr2) / (d1 - d2)
          end if
          if (abs(derv) > DEM30) then
            dlh = -fhstr1 / derv
          else
            dlh = DZERO
          end if
          dpp = d1 + dlh
          !
          ! -- updated depth outside of endpoints - use bisection instead
          if ((dpp >= en2) .or. (dpp <= en1)) then
            if (abs(dlh) > abs(dlhold) .or. dpp < DEM30) then
              ibflg = 1
              dpp = DHALF * (en1 + en2)
            end if
          end if
          !
          ! -- check for slow convergence
          ! -- set flags to determine if the Newton-Raphson method oscillates
          !    or if convergence is slow
          if (qgwf1*qgwfold < DEM30) then
            iic2 = iic2 + 1
          else
            iic2 = 0
          end if
          if (qgwf1 < DEM30) then
            iic3 = iic3 + 1
          else
            iic3 = 0
          end if
          if (dlh*dlhold < DEM30 .or. ABS(dlh) > ABS(dlhold)) then
            iic = iic + 1
          end if
          iic4 = 0
          if (iic3 > 7 .and. iic > 12) then
            iic4 = 1
          end if
          !
          ! -- switch to bisection when the Newton-Raphson method oscillates
          !    or when convergence is slow
          if (iic2 > 7 .or. iic > 12 .or. iic4 == 1) then
            ibflg = 1
            dpp = DHALF * (en1 + en2)
          end if
          !
          ! -- Calculate perturbed gwf flow
          call this%sfr_calc_qgwf(n, dpp, hgwf, qgwfp)
          qgwfp = -qgwfp
          if (qgwfp > qsrc) then
            qgwfp = qsrc
            if (abs(en1-en2) < this%dmaxchg*DEM6) then
              call this%sfr_rectch_depth(n, (qmpsrc-DHALF*qgwfp), dpp)
            end if
          end if
          call this%sfr_rectch_depth(n, (qmpsrc-DHALF*qgwfp), dx)
        end if
        !
        ! --
        fp = dpp - dx
        if (ibflg == 1) then
          dlh = fp
          ! -- change end points
          ! -- root is between f1 and fp
          if (f1*fp < DZERO) then
            en2 = dpp
            f2 = fp
          ! -- root is between fp and f2
          else
            en1 = dpp
            f1 = fp
          end if
          err = min(abs(fp), abs(en2-en1))
        else
          err = abs(dlh)
        end if
        !
        ! -- check for convergence and exit if converged
        if (err < this%dmaxchg) then
          d1 = dpp
          qgwf = qgwfp
          qd = qsrc - qgwf
          exit
        end if
        !
        ! -- save iterates
        errold = err
        dlhold = dlh
        if (ibflg == 1) then
          qgwfold = qgwfp
        else
          qgwfold = qgwf1
        end if
      !
      ! -- end of iteration
      end do
    end if itersol

    ! -- simple routing option or where depth = 0 and hgwf < bt
    if (isolve == 0) then
      call this%sfr_calc_qgwf(n, d1, hgwf, qgwf)
      qgwf = -qgwf
      !
      ! -- leakage exceeds inflow
      if (qgwf > qsrc) then
        d1 = DZERO
        call this%sfr_calc_qsource(n, d1, qsrc)
        qgwf = qsrc
      end if
      ! -- set qd
      qd = qsrc - qgwf
    end if
    !
    ! -- update sfr stage
    hsfr = tp + d1
    !
    ! -- update stored values
    if (lupdate) then
      !
      ! -- save depth and calculate stage
      this%depth(n) = d1
      this%stage(n) = hsfr
      !
      ! -- update flows
      call this%sfr_update_flows(n, qd, qgwf)
    end if
    !
    ! -- calculate sumleak and sumrch
    sumleak = DZERO
    sumrch = DZERO
    if (this%gwfiss == 0) then
      sumleak = qgwf
    else
      sumleak = qgwf
    end if
    if (hgwf < bt) then
      sumrch = qgwf
    end if
    !
    ! -- make final qgwf calculation and obtain
    !    gwfhcof and gwfrhs values
    call this%sfr_calc_qgwf(n, d1, hgwf, qgwf, gwfhcof, gwfrhs)
    !
    !
    if (abs(sumleak) > DZERO) then
      ! -- stream leakage is not head dependent
      if (hgwf < bt) then
        rhs = rhs - sumrch
      !
      ! -- stream leakage is head dependent
      else if ((sumleak-qsrc) < -DEM30) then
        if (this%gwfiss == 0) then
          rhs = rhs + gwfrhs - sumrch
        else
          rhs = rhs + gwfrhs
        end if
        hcof = gwfhcof
      !
      ! -- place holder for UZF
      else
        if (this%gwfiss == 0) then
          rhs = rhs - sumleak - sumrch
        else
          rhs = rhs - sumleak
        end if
      end if
    !
    ! -- add groundwater leakage
    else if (hgwf < bt) then
      rhs = rhs - sumrch
    end if
    !
    ! -- return
    return
  end subroutine sfr_solve

  subroutine sfr_update_flows(this, n, qd, qgwf)
  ! ******************************************************************************
  ! sfr_update_flows -- Update downstream and groundwater leakage terms for reach
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType), intent(inout) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(inout) :: qd
      real(DP), intent(in) :: qgwf
      ! -- local
      integer(I4B) :: i
      integer(I4B) :: n2
      integer(I4B) :: idiv
      integer(I4B) :: jpos
      real(DP) :: qdiv
      real(DP) :: f
  ! ------------------------------------------------------------------------------
    !
    ! -- update reach terms
    !
    ! -- save final downstream stream flow
    this%dsflow(n) = qd
    !
    ! -- save groundwater leakage
    this%gwflow(n) = qgwf
    !
    ! -- route downstream flow
    if (qd > DZERO) then
      !
      ! -- route water to diversions
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        if (this%idir(i) > 0) cycle
        idiv = this%idiv(i)
        if (idiv == 0) cycle
        jpos = this%iadiv(n) + idiv - 1
        call this%sfr_calc_div(n, idiv, qd, qdiv)
        this%qconn(i) = qdiv
        this%divq(jpos) = qdiv
      end do
      !
      ! -- Mover terms: store outflow after diversion loss
      !    as qformvr and reduce outflow (qd)
      !    by how much was actually sent to the mover
      if (this%imover == 1) then
        call this%pakmvrobj%accumulate_qformvr(n, qd)
        qd = MAX(qd - this%pakmvrobj%get_qtomvr(n), DZERO)
      endif
      !
      ! -- route remaining water to downstream reaches
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        if (this%idir(i) > 0) cycle
        if (this%idiv(i) > 0) cycle
        n2 = this%ja(i)
        f = this%ustrf(n2) / this%ftotnd(n)
        this%qconn(i) = qd * f
      end do
    else
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        if (this%idir(i) > 0) cycle
        this%qconn(i) = DZERO
      end do
    end if
    !
    ! -- return
    return
  end subroutine sfr_update_flows

  subroutine sfr_calc_qd(this, n, depth, hgwf, qgwf, qd)
  ! ******************************************************************************
  ! sfr_calc_dq -- Calculate downstream flow for reach
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(in) :: depth
      real(DP), intent(in) :: hgwf
      real(DP), intent(inout) :: qgwf
      real(DP), intent(inout) :: qd
      ! -- local
      real(DP) :: qsrc
  ! ------------------------------------------------------------------------------
    !
    ! -- initialize residual
    qd = DZERO
    !
    ! -- calculate total water sources excluding groundwater leakage
    call this%sfr_calc_qsource(n, depth, qsrc)
    !
    ! -- estimate groundwater leakage
    call this%sfr_calc_qgwf(n, depth, hgwf, qgwf)
    if (-qgwf > qsrc) qgwf = -qsrc
    !
    ! -- calculate down stream flow
    qd = qsrc + qgwf
    !
    ! -- limit downstream flow to a positive value
    if (qd < DEM30) qd = DZERO
    !
    ! -- return
    return
  end subroutine sfr_calc_qd

  subroutine sfr_calc_qsource(this, n, depth, qsrc)
  ! ******************************************************************************
  ! sfr_calc_qsource -- Calculate sum of sources for reach - excluding
  !                     reach leakage
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(in) :: depth
      real(DP), intent(inout) :: qsrc
      ! -- local
      real(DP) :: qu, qi, qr, qe, qro, qfrommvr
      real(DP) :: qt
      real(DP) :: a, ae
  ! ------------------------------------------------------------------------------
    !
    ! -- initialize residual
    qsrc = DZERO
    !
    ! -- calculate flow terms
    qu = this%usflow(n)
    qi = this%inflow(n)
    qro = this%runoff(n)
    !
    ! -- calculate rainfall and evap
    a = this%surface_area(n)
    ae = this%surface_area_wet(n, depth)
    qr = this%rain(n) * a
    qe = this%evap(n) * a
    !
    ! -- calculate mover term
    qfrommvr = DZERO
    if (this%imover == 1) then
      qfrommvr = this%pakmvrobj%get_qfrommvr(n)
    endif
    !
    ! -- calculate down stream flow
    qsrc = qu + qi + qr - qe + qro + qfrommvr
    !
    ! -- adjust runoff or evaporation if sum of sources is negative
    if (qsrc < DZERO) then
      !
      ! -- calculate sources without et
      qt = qu + qi + qr + qro + qfrommvr
      !
      ! -- runoff exceeds sources of water for reach
      if (qt < DZERO) then
        qro = -(qu + qi + qr + qfrommvr)
        qe = DZERO
      !
      ! -- evaporation exceeds sources of water for reach
      else
        qe = qu + qi + qr + qro + qfrommvr
      end if
      qsrc = qu + qi + qr - qe + qro + qfrommvr
    end if
    !
    ! -- return
    return
  end subroutine sfr_calc_qsource


  subroutine sfr_calc_qman(this, n, depth, qman)
  ! ******************************************************************************
  ! sfr_calc_qman -- Calculate stream flow using Manning's equation
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(in) :: depth
      real(DP), intent(inout) :: qman
      ! -- local
      real(DP) :: sat
      real(DP) :: derv
      real(DP) :: s, r, aw, wp, rh
  ! ------------------------------------------------------------------------------
    !
    ! -- initialize qman
    qman = DZERO
    !
    ! -- calculate terms for Manning's equation
    call sChSmooth(depth, sat, derv)
    s = this%slope(n)
    r = this%rough(n)
    aw = this%area_wet(n, depth)
    wp = this%perimeter_wet(n)
    rh = DZERO
    if (wp > DZERO) then
      rh = aw / wp
    end if
    !
    ! -- calculate flow
    qman = sat * this%unitconv * aw * (rh**DTWOTHIRDS) * sqrt(s) / r
    !
    ! -- return
    return
  end subroutine sfr_calc_qman


  subroutine sfr_calc_qgwf(this, n, depth, hgwf, qgwf, gwfhcof, gwfrhs)
  ! ******************************************************************************
  ! sfr_calc_qgwf -- Calculate sfr-aquifer exchange (relative to sfr reach)
  !   so flow is positive into the stream reach
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(in) :: depth
      real(DP), intent(in) :: hgwf
      real(DP), intent(inout) :: qgwf
      real(DP), intent(inout), optional :: gwfhcof
      real(DP), intent(inout), optional :: gwfrhs
      ! -- local
      integer(I4B) :: node
      real(DP) :: tp
      real(DP) :: bt
      real(DP) :: hsfr
      real(DP) :: htmp
      real(DP) :: cond
      real(DP) :: sat
      real(DP) :: derv
      real(DP) :: gwfhcof0, gwfrhs0
  ! ------------------------------------------------------------------------------
    !
    ! -- initialize qgwf
    qgwf = DZERO
    !
    ! -- skip sfr-aquifer exchange in external cells
    node = this%igwfnode(n)
    if (node < 1) return
    !
    ! -- skip sfr-aquifer exchange in inactive cells
    if (this%ibound(node) == 0) return
    !
    ! -- calculate saturation
    call sChSmooth(depth, sat, derv)
    !
    ! -- calculate conductance
    call this%sfr_calc_cond(n, cond)
    !
    ! -- calculate groundwater leakage
    tp = this%strtop(n)
    bt = tp - this%bthick(n)
    hsfr = tp + depth
    htmp = hgwf
    if (htmp < bt) then
      htmp = bt
    end if
    qgwf = sat * cond * (htmp - hsfr)
    gwfrhs0 = -sat * cond * hsfr
    gwfhcof0 = -sat * cond
    !
    ! Add density contributions, if active
    if (this%idense /= 0) then
      call this%sfr_calculate_density_exchange(n, hsfr, hgwf, cond, tp,        &
                                               qgwf, gwfhcof0, gwfrhs0)
    end if
    !
    ! -- Set gwfhcof and gwfrhs if present
    if (present(gwfhcof)) gwfhcof = gwfhcof0
    if (present(gwfrhs)) gwfrhs = gwfrhs0
    !
    ! -- return
    return
  end subroutine sfr_calc_qgwf

  subroutine sfr_calc_cond(this, n, cond)
  ! ******************************************************************************
  ! sfr_calc_qgwf -- Calculate sfr-aquifer exchange
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      real(DP), intent(inout) :: cond
      ! -- local
      integer(I4B) :: node
      real(DP) :: wp
  ! ------------------------------------------------------------------------------
    !
    ! -- initialize a few variables
    cond = DZERO
    node = this%igwfnode(n)
    if (node > 0) then
      if (this%ibound(node) > 0) then
        wp = this%perimeter_wet(n)
        cond = this%hk(n) * this%length(n) * wp / this%bthick(n)
      end if
    end if
    !
    ! -- return
    return
  end subroutine sfr_calc_cond


  subroutine sfr_calc_div(this, n, i, qd, qdiv)
  ! ******************************************************************************
  ! sfr_calc_div -- Calculate the diversion flow for reach
  ! ******************************************************************************
  !
  !    SPECIFICATIONS:
  ! ------------------------------------------------------------------------------
      class(SfrType) :: this
      integer(I4B), intent(in) :: n
      integer(I4B), intent(in) :: i
      real(DP), intent(inout) :: qd
      real(DP), intent(inout) :: qdiv
      ! -- local
      character (len=10) :: cp
      integer(I4B) :: jpos
      integer(I4B) :: n2
      !integer(I4B) :: ip
      real(DP) :: v
  ! ------------------------------------------------------------------------------
    !
    ! -- set local variables
    jpos = this%iadiv(n) + i - 1
    n2 = this%divreach(jpos)
    cp = this%divcprior(jpos)
    v = this%divflow(jpos)
    !
    ! -- calculate diversion
    select case(cp)
      ! -- flood diversion
      case ('EXCESS')
        if (qd < v) then
          v = DZERO
        else
          v = qd - v
        end if
      ! -- diversion percentage
      case ('FRACTION')
        v = qd * v
      ! -- STR priority algorithm
      case ('THRESHOLD')
        if (qd < v) then
          v = DZERO
        end if
      ! -- specified diversion
      case ('UPTO')
        if (v > qd) then
          v = qd
        end if
      case default
        v = DZERO
    end select
    !
    ! -- update upstream from for downstream reaches
    qd = qd - v
    qdiv = v
    !
    ! -- return
    return
  end subroutine sfr_calc_div

  subroutine sfr_rectch_depth(this, n, q1, d1)
    class(SfrType) :: this
    integer(I4B), intent(in) :: n
    real(DP), intent(in) :: q1
    real(DP), intent(inout) :: d1
    ! -- local
    real(DP) :: w
    real(DP) :: s
    real(DP) :: r
    real(DP) :: qconst
    ! -- code
    ! -- calculate stream depth at the midpoint
    w = this%width(n)
    s = this%slope(n)
    r = this%rough(n)
    qconst = this%unitconv * w * sqrt(s) / r
    d1 = (q1 / qconst)**DP6
    if (d1 < DEM30) d1 = DZERO
    ! -- return
    return
  end subroutine sfr_rectch_depth


  subroutine sfr_check_reaches(this)
    class(SfrType) :: this
    ! -- local
    character (len= 5) :: crch
    character (len=10) :: cval
    character (len=30) :: nodestr
    character (len=LINELENGTH) :: title
    character (len=LINELENGTH) :: text
    integer(I4B) :: n, nn
    real(DP) :: btgwf, bt
    ! -- code
    !
    ! -- setup inputtab tableobj
    if (this%iprpak /= 0) then
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') STATIC REACH DATA'
      call table_cr(this%inputtab, this%packName, title)
      call this%inputtab%table_df(this%maxbound, 10, this%iout)
      text = 'NUMBER'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      text = 'CELLID'
      call this%inputtab%initialize_column(text, 20, alignment=TABLEFT)
      text = 'LENGTH'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'WIDTH'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'SLOPE' 
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'TOP'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'THICKNESS'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'HK'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'ROUGHNESS'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      text = 'UPSTREAM FRACTION'
      call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
    end if
    !
    ! -- check the reach data for simple errors
    do n = 1, this%maxbound
      write(crch, '(i5)') n
      nn = this%igwfnode(n)
      if (nn > 0) then
        btgwf = this%dis%bot(nn)
        call this%dis%noder_to_string(nn, nodestr)
      else
        nodestr = 'none'
      end if
      ! -- check reach length
      if (this%length(n) <= DZERO) then
        errmsg = 'Reach ' // crch // ' length must be greater than 0.0.'
        call store_error(errmsg)
      end if
      ! -- check reach width
      if (this%width(n) <= DZERO) then
        errmsg = 'Reach ' // crch // ' width must be greater than 0.0.'
        call store_error(errmsg)
      end if
      ! -- check reach slope
      if (this%slope(n) <= DZERO) then
        errmsg = 'Reach ' // crch // ' slope must be greater than 0.0.'
        call store_error(errmsg)
      end if
      ! -- check bed thickness and bed hk for reaches connected to GWF
      if (nn > 0) then
        bt = this%strtop(n) - this%bthick(n)
        if (bt <= btgwf .and. this%icheck /= 0) then
          write(cval,'(f10.4)') bt
          errmsg = 'Reach ' // crch // ' bed bottom (rtp-rbth =' //              &
                   cval // ') must be greater than the bottom of cell (' //      &
                   nodestr
          write(cval,'(f10.4)') btgwf
          errmsg = trim(adjustl(errmsg)) // '=' // cval // ').'
          call store_error(errmsg)
        end if
        if (this%hk(n) < DZERO) then
          errmsg = 'Reach ' // crch // ' hk must be greater than or equal to 0.0.'
          call store_error(errmsg)
        end if
      end if
      ! -- check reach roughness
      if (this%rough(n) <= DZERO) then
        errmsg = 'Reach ' // crch // " Manning's roughness " //                  &
                 'coefficient must be greater than 0.0.'
        call store_error(errmsg)
      end if
      ! -- check reach upstream fraction
      if (this%ustrf(n) < DZERO) then
        errmsg = 'Reach ' // crch // ' upstream fraction must be greater ' //    &
                 'than or equal to 0.0.'
        call store_error(errmsg)
      end if
      ! -- write summary of reach information
      if (this%iprpak /= 0) then
        call this%inputtab%add_term(n)
        call this%inputtab%add_term(nodestr)
        call this%inputtab%add_term(this%length(n))
        call this%inputtab%add_term(this%width(n))
        call this%inputtab%add_term(this%slope(n))
        call this%inputtab%add_term(this%strtop(n))
        call this%inputtab%add_term(this%bthick(n))
        call this%inputtab%add_term(this%hk(n))
        call this%inputtab%add_term(this%rough(n))
        call this%inputtab%add_term(this%ustrf(n))
      end if
    end do

    ! -- return
    return
  end subroutine sfr_check_reaches


  subroutine sfr_check_connections(this)
    class(SfrType) :: this
    ! -- local
    character (len= 5) :: crch
    character (len= 5) :: crch2
    character (len=LINELENGTH) :: text
    character (len=LINELENGTH) :: title
    integer(I4B) :: n, nn, nc
    integer(I4B) :: i, ii
    integer(I4B) :: ifound
    integer(I4B) :: ierr
    integer(I4B) :: maxconn
    integer(I4B) :: ntabcol
    ! -- code
    !
    ! -- create input table for reach connections data
    if (this%iprpak /= 0) then
      !
      ! -- calculate the maximum number of connections
      maxconn = 0
      do n = 1, this%maxbound
        maxconn = max(maxconn, this%nconnreach(n))
      end do
      ntabcol = 1 + maxconn
      !
      ! -- reset the input table object
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') STATIC REACH CONNECTION DATA'
      call table_cr(this%inputtab, this%packName, title)
      call this%inputtab%table_df(this%maxbound, ntabcol, this%iout)
      text = 'REACH'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      do n = 1, maxconn
        write(text, '(a,1x,i6)') 'CONN', n
        call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      end do
    end if
    !
    ! -- check the reach connections for simple errors
    ! -- connection check
    do n = 1, this%maxbound
      write(crch, '(i5)') n
      eachconn: do i = this%ia(n) + 1, this%ia(n+1) - 1
        nn = this%ja(i)
        write(crch2, '(i5)') nn
        ifound = 0
        connreach: do ii = this%ia(nn) + 1, this%ia(nn+1) - 1
          nc = this%ja(ii)
          if (nc == n) then
            ifound = 1
            exit connreach
          end if
        end do connreach
        if (ifound /= 1) then
          errmsg = 'Reach ' // crch // ' is connected to ' //                    &
                   'reach ' // crch2 // ' but reach ' // crch2 //                &
                   ' is not connected to reach ' // crch // '.'
          call store_error(errmsg)
        end if
      end do eachconn
      !
      ! -- write connection data to the table
      if (this%iprpak /= 0) then
        call this%inputtab%add_term(n)
        do i = this%ia(n) + 1, this%ia(n+1) - 1
          call this%inputtab%add_term(this%ja(i))
        end do
        nn = maxconn - this%nconnreach(n)
        do i = 1, nn
          call this%inputtab%add_term(' ')
        end do
      end if
    end do
    !
    ! -- check for incorrect connections between upstream connections
    !
    ! -- check upstream connections for each reach
    ierr = 0
    do n = 1, this%maxbound
      write(crch, '(i5)') n
      eachconnv: do i = this%ia(n) + 1, this%ia(n + 1) - 1
        !
        ! -- skip downstream connections
        if (this%idir(i) < 0) cycle eachconnv
        nn = this%ja(i)
        write(crch2, '(i5)') nn
        connreachv: do ii = this%ia(nn) + 1, this%ia(nn+1) - 1
          ! -- skip downstream connections
          if (this%idir(ii) < 0) cycle connreachv
          nc = this%ja(ii)
          !
          ! -- if nc == n then that means reach n is an upstream connection for
          !    reach nn and reach nn is an upstream connection for reach n
          if (nc == n) then
            ierr = ierr + 1
            errmsg = 'Reach ' // crch // ' is connected to ' //                  &
                     'reach ' // crch2 // ' but streamflow from reach ' //       &
                     crch // ' to reach ' // crch2 // ' is not permitted.'
            call store_error(errmsg)
            exit connreachv
          end if
        end do connreachv
      end do eachconnv
    end do
    !
    ! -- terminate if connectivity errors
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- check that downstream reaches for a reach are
    !    the upstream reaches for the reach
    do n = 1, this%maxbound
      write(crch, '(i5)') n
      eachconnds: do i = this%ia(n) + 1, this%ia(n+1) - 1
        nn = this%ja(i)
        if (this%idir(i) > 0) cycle eachconnds
        write(crch2, '(i5)') nn
        ifound = 0
        connreachds: do ii = this%ia(nn) + 1, this%ia(nn+1) - 1
          nc = this%ja(ii)
          if (nc == n) then
            if (this%idir(i) /= this%idir(ii)) then
              ifound = 1
            end if
            exit connreachds
          end if
        end do connreachds
        if (ifound /= 1) then
          errmsg = 'Reach ' // crch // ' downstream connected reach ' //         &
                   'is reach ' // crch2 // ' but reach ' // crch // ' is not' // &
                   ' the upstream connected reach for reach ' // crch2 // '.'
          call store_error(errmsg)
        end if
      end do eachconnds
    end do
    !
    ! -- create input table for upstream and downstream connections
    if (this%iprpak /= 0) then
      !
      ! -- calculate the maximum number of upstream connections
      maxconn = 0
      do n = 1, this%maxbound
        ii = 0
        do i = this%ia(n) + 1, this%ia(n+1) - 1
          if (this%idir(i) > 0) then
            ii = ii + 1
          end if
        end do
        maxconn = max(maxconn, ii)
      end do
      ntabcol = 1 + maxconn
      !
      ! -- reset the input table object
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') STATIC UPSTREAM REACH ' //           &
              'CONNECTION DATA'
      call table_cr(this%inputtab, this%packName, title)
      call this%inputtab%table_df(this%maxbound, ntabcol, this%iout)
      text = 'REACH'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      do n = 1, maxconn
        write(text, '(a,1x,i6)') 'UPSTREAM CONN', n
        call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      end do
      !
      ! -- upstream connection data
      do n = 1, this%maxbound
        call this%inputtab%add_term(n)
        ii = 0
        do i = this%ia(n) + 1, this%ia(n+1) - 1
          if (this%idir(i) > 0) then
            call this%inputtab%add_term(this%ja(i))
            ii = ii + 1
          end if
        end do
        nn = maxconn - ii
        do i = 1, nn
          call this%inputtab%add_term(' ')
        end do  
      end do
      !
      ! -- calculate the maximum number of downstream connections
      maxconn = 0
      do n = 1, this%maxbound
        ii = 0
        do i = this%ia(n) + 1, this%ia(n+1) - 1
          if (this%idir(i) < 0) then
            ii = ii + 1
          end if
        end do
        maxconn = max(maxconn, ii)
      end do
      ntabcol = 1 + maxconn
      !
      ! -- reset the input table object
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') STATIC DOWNSTREAM ' //               &
              'REACH CONNECTION DATA'
      call table_cr(this%inputtab, this%packName, title)
      call this%inputtab%table_df(this%maxbound, ntabcol, this%iout)
      text = 'REACH'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      do n = 1, maxconn
        write(text, '(a,1x,i6)') 'DOWNSTREAM CONN', n
        call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      end do
      !
      ! -- downstream connection data
      do n = 1, this%maxbound
        call this%inputtab%add_term(n)
        ii = 0
        do i = this%ia(n) + 1, this%ia(n+1) - 1
          if (this%idir(i) < 0) then
            call this%inputtab%add_term(this%ja(i))
            ii = ii + 1
          end if
        end do
        nn = maxconn - ii
        do i = 1, nn
          call this%inputtab%add_term(' ')
        end do  
      end do
    end if
    !
    ! -- return
    return
  end subroutine sfr_check_connections


  subroutine sfr_check_diversions(this)
    class(SfrType) :: this
    ! -- local
    character (len=LINELENGTH) :: title
    character (len=LINELENGTH) :: text
    character (len= 5) :: crch
    character (len= 5) :: cdiv
    character (len= 5) :: crch2
    character (len=10) :: cprior
    integer(I4B) :: maxdiv
    integer(I4B) :: n
    integer(I4B) :: nn
    integer(I4B) :: nc
    integer(I4B) :: ii
    integer(I4B) :: idiv
    integer(I4B) :: ifound
    integer(I4B) :: jpos
    ! -- format
10  format('Diversion ',i0,' of reach ',i0,                                      &
           ' is invalid or has not been defined.')
    ! -- code
    !
    ! -- write header
    if (this%iprpak /= 0) then
      !
      ! -- determine the maximum number of diversions
      maxdiv = 0
      do n = 1, this%maxbound
        maxdiv = maxdiv + this%ndiv(n)
      end do
      !
      ! -- reset the input table object
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') REACH DIVERSION DATA'
      call table_cr(this%inputtab, this%packName, title)
      call this%inputtab%table_df(maxdiv, 4, this%iout)
      text = 'REACH'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      text = 'DIVERSION'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      text = 'REACH 2'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      text = 'CPRIOR'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
    end if
    !
    ! -- check that diversion data are correct
    do n = 1, this%maxbound
      if (this%ndiv(n) < 1) cycle
      write(crch, '(i5)') n
      
      do idiv = 1, this%ndiv(n)
        !
        ! -- determine diversion index
        jpos = this%iadiv(n) + idiv - 1
        !
        ! -- write idiv to cdiv
        write(cdiv, '(i5)') idiv
        !
        !
        nn = this%divreach(jpos)
        write(crch2, '(i5)') nn
        !
        ! -- make sure diversion reach is connected to current reach
        ifound = 0
        if (nn < 1 .or. nn > this%maxbound) then
          write(errmsg,10) idiv, n
          call store_error(errmsg)
          cycle
        end if
        connreach: do ii = this%ia(nn) + 1, this%ia(nn+1) - 1
          nc = this%ja(ii)
          if (nc == n) then
            if (this%idir(ii) > 0) then
              ifound = 1
            end if
            exit connreach
          end if
        end do connreach
        if (ifound /= 1) then
          errmsg = 'Reach ' // crch // ' is not a upstream reach for ' //        &
                   'reach ' // crch2 // ' as a result diversion ' // cdiv //     &
                   ' from reach ' // crch //' to reach ' // crch2 //             &
                   ' is not possible. Check reach connectivity.'
          call store_error(errmsg)
        end if
        ! -- iprior
        cprior = this%divcprior(jpos)
        !
        ! -- add terms to the table
        if (this%iprpak /= 0) then
          call this%inputtab%add_term(n)
          call this%inputtab%add_term(idiv)
          call this%inputtab%add_term(nn)
          call this%inputtab%add_term(cprior)
        end if
      end do
    end do
    !
    ! -- return
    return
  end subroutine sfr_check_diversions


  subroutine sfr_check_ustrf(this)
    class(SfrType) :: this
    ! -- local
    character (len=LINELENGTH) :: title
    character (len=LINELENGTH) :: text
    logical :: lcycle
    logical :: ladd
    character (len=5) :: crch, crch2
    character (len=10) :: cval
    integer(I4B) :: maxcols
    integer(I4B) :: npairs
    integer(I4B) :: ipair
    integer(I4B) :: i
    integer(I4B) :: n
    integer(I4B) :: n2
    integer(I4B) :: idiv
    integer(I4B) :: i0
    integer(I4B) :: i1
    integer(I4B) :: jpos
    integer(I4B) :: ids
    real(DP) :: f
    real(DP) :: rval
    ! -- code
    !
    ! -- write table header
    if (this%iprpak /= 0) then
      !
      ! -- determine the maximum number of columns
      npairs = 0
      do n = 1, this%maxbound
        ipair = 0
        ec: do i = this%ia(n) + 1, this%ia(n+1) - 1
          !
          ! -- skip upstream connections
          if (this%idir(i) > 0) cycle ec
          n2 = this%ja(i)
          !
          ! -- skip inactive downstream reaches
          if (this%iboundpak(n2) == 0) cycle ec
          !
          ! -- increment ipair and see if it exceeds npairs
          ipair = ipair + 1
          npairs = max(npairs, ipair)
        end do ec
      end do
      maxcols = 1 + npairs * 2
      !
      ! -- reset the input table object
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') CONNECTED REACH UPSTREAM '        // &
              'FRACTION DATA'
      call table_cr(this%inputtab, this%packName, title)
      call this%inputtab%table_df(this%maxbound, maxcols, this%iout)
      text = 'REACH'
      call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
      do i = 1, npairs
        write(cval, '(i10)') i
        text = 'DOWNSTREAM REACH ' // trim(adjustl(cval)) 
        call this%inputtab%initialize_column(text, 10, alignment=TABCENTER)
        text = 'FRACTION ' // trim(adjustl(cval)) 
        call this%inputtab%initialize_column(text, 12, alignment=TABCENTER)
      end do
    end if
    !
    ! -- fill diversion number for each connection
    do n = 1, this%maxbound
      do idiv = 1, this%ndiv(n)
        i0 = this%iadiv(n)
        i1 = this%iadiv(n+1) - 1
        do jpos = i0, i1 
          do i = this%ia(n) + 1, this%ia(n+1) - 1
            n2 = this%ja(i)
            if (this%divreach(jpos) == n2) then
              this%idiv(i) = jpos - i0 + 1
              exit
            end if 
          end do
        end do
      end do
    end do
    !
    ! -- check that the upstream fraction for reach connected by
    !    a diversion is zero
    do n = 1, this%maxbound
      !
      ! -- determine the number of downstream reaches
      ids = 0
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        if (this%idir(i) < 0) then
          ids = ids + 1
        end if
      end do
      !
      ! -- evaluate the diversions
      do idiv = 1, this%ndiv(n)
        jpos = this%iadiv(n) + idiv - 1
        n2 = this%divreach(jpos)
        f = this%ustrf(n2)
        if (f /= DZERO) then
          write(errmsg, '(a,2(1x,i0,1x,a),1x,a,g0,a,2(1x,a))')                   &
            'Reach', n, 'is connected to reach', n2, 'by a diversion',           &
            'but the upstream fraction is not equal to zero (', f, '). Check',   &
            trim(this%packName), 'package diversion and package data.'
          if (ids > 1) then
            call store_error(errmsg)
          else
            write(warnmsg, '(a,3(1x,a))')                                        &
              trim(warnmsg), 'A warning instead of an error is issued because',  &
              'the reach is only connected to the diversion reach in the ',      &
              'downstream direction.'
            call store_warning(warnmsg)
          end if
        end if
      end do
    end do
    !
    ! -- calculate the total fraction of connected reaches that are
    !    not diversions and check that the sum of upstream fractions
    !    is equal to 1 for each reach
    do n = 1, this%maxbound
      ids = 0
      rval = DZERO
      f = DZERO
      write(crch, '(i5)') n
      if (this%iprpak /= 0) then
        call this%inputtab%add_term(n)
      end if
      ipair = 0
      eachconn: do i = this%ia(n) + 1, this%ia(n+1) - 1
        lcycle = .FALSE.
        !
        ! -- initialize downstream connection q
        this%qconn(i) = DZERO
        !
        ! -- skip upstream connections
        if (this%idir(i) > 0) then
          lcycle = .TRUE.
        end if
        n2 = this%ja(i)
        !
        ! -- skip inactive downstream reaches
        if (this%iboundpak(n2) == 0) then
          lcycle = .TRUE.
        end if
        if (lcycle) then
          cycle eachconn
        end if
        ipair = ipair + 1
        write(crch2, '(i5)') n2
        ids = ids + 1
        ladd = .true.
        f = f + this%ustrf(n2)
        write(cval, '(f10.4)') this%ustrf(n2)
        !
        ! -- write upstream fractions
        if (this%iprpak /= 0) then
          call this%inputtab%add_term(n2)
          call this%inputtab%add_term(this%ustrf(n2))
        end if
        eachdiv: do idiv = 1, this%ndiv(n)
          jpos = this%iadiv(n) + idiv - 1
          if (this%divreach(jpos) == n2) then
            ladd = .false.
            exit eachdiv
          end if
        end do eachdiv
        if (ladd) then
          rval = rval + this%ustrf(n2)
        end if
      end do eachconn
      this%ftotnd(n) = rval
      !
      ! -- write remaining table columns
      if (this%iprpak /= 0) then
        ipair = ipair + 1
        do i = ipair, npairs
          call this%inputtab%add_term('  ')
          call this%inputtab%add_term('  ')
        end do
      end if
      !
      ! -- evaluate if an error condition has occured
      !    the sum of fractions is not equal to 1
      if (ids /= 0) then
        if (abs(f-DONE) > DEM6) then
          write(errmsg, '(a,1x,i0,1x,a,g0,a,3(1x,a))')                           &
            'Upstream fractions for reach ', n, 'is not equal to one (', f,      &
            '). Check', trim(this%packName), 'package reach connectivity and',       &
            'package data.'
          call store_error(errmsg)
        end if
      end if
    end do
    !
    ! -- return
    return
  end subroutine sfr_check_ustrf

  subroutine sfr_setup_budobj(this)
! ******************************************************************************
! sfr_setup_budobj -- Set up the budget object that stores all the sfr flows
!   The terms listed here must correspond in number and order to the ones 
!   listed in the sfr_fill_budobj routine.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LENBUDTXT
    ! -- dummy
    class(SfrType) :: this
    ! -- local
    integer(I4B) :: nbudterm
    integer(I4B) :: i, n, n1, n2
    integer(I4B) :: maxlist, naux
    integer(I4B) :: idx
    real(DP) :: q
    character(len=LENBUDTXT) :: text
    character(len=LENBUDTXT), dimension(1) :: auxtxt
! ------------------------------------------------------------------------------
    !
    ! -- Determine the number of sfr budget terms. These are fixed for 
    !    the simulation and cannot change.  This includes FLOW-JA-FACE
    !    so they can be written to the binary budget files, but these internal
    !    flows are not included as part of the budget table.
    nbudterm = 8
    if (this%imover == 1) nbudterm = nbudterm + 2
    if (this%naux > 0) nbudterm = nbudterm + 1
    !
    ! -- set up budobj
    call budgetobject_cr(this%budobj, this%packName)
    call this%budobj%budgetobject_df(this%maxbound, nbudterm, 0, 0)
    idx = 0
    !
    ! -- Go through and set up each budget term
    text = '    FLOW-JA-FACE'
    idx = idx + 1
    maxlist = this%nconn
    naux = 1
    auxtxt(1) = '       FLOW-AREA'
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux, auxtxt)
    !
    ! -- store connectivity
    call this%budobj%budterm(idx)%reset(this%nconn)
    q = DZERO
    do n = 1, this%maxbound
      n1 = n
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        n2 = this%ja(i)
        call this%budobj%budterm(idx)%update_term(n1, n2, q)
      end do
    end do
    !
    ! -- 
    text = '             GWF'
    idx = idx + 1
    maxlist = this%maxbound - this%ianynone
    naux = 1
    auxtxt(1) = '       FLOW-AREA'
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%name_model, &
                                             maxlist, .false., .true., &
                                             naux, auxtxt)
    call this%budobj%budterm(idx)%reset(this%maxbound)
    q = DZERO
    do n = 1, this%maxbound
      n2 = this%igwfnode(n)
      if (n2 > 0) then
        call this%budobj%budterm(idx)%update_term(n, n2, q)
      end if
    end do
    !
    ! -- 
    text = '        RAINFALL'
    idx = idx + 1
    maxlist = this%maxbound
    naux = 0
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux)
    !
    ! -- 
    text = '     EVAPORATION'
    idx = idx + 1
    maxlist = this%maxbound
    naux = 0
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux)
    !
    ! -- 
    text = '          RUNOFF'
    idx = idx + 1
    maxlist = this%maxbound
    naux = 0
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux)
    !
    ! -- 
    text = '      EXT-INFLOW'
    idx = idx + 1
    maxlist = this%maxbound
    naux = 0
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux)
    !
    ! -- 
    text = '     EXT-OUTFLOW'
    idx = idx + 1
    maxlist = this%maxbound
    naux = 0
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux)
    !
    ! -- 
    text = '         STORAGE'
    idx = idx + 1
    maxlist = this%maxbound
    naux = 1
    auxtxt(1) = '          VOLUME'
    call this%budobj%budterm(idx)%initialize(text, &
                                             this%name_model, &
                                             this%packName, &
                                             this%name_model, &
                                             this%packName, &
                                             maxlist, .false., .false., &
                                             naux, auxtxt)
    !
    ! -- 
    if (this%imover == 1) then
      !
      ! -- 
      text = '        FROM-MVR'
      idx = idx + 1
      maxlist = this%maxbound
      naux = 0
      call this%budobj%budterm(idx)%initialize(text, &
                                               this%name_model, &
                                               this%packName, &
                                               this%name_model, &
                                               this%packName, &
                                               maxlist, .false., .false., &
                                               naux)
      !
      ! -- 
      text = '          TO-MVR'
      idx = idx + 1
      maxlist = this%maxbound
      naux = 0
      call this%budobj%budterm(idx)%initialize(text, &
                                               this%name_model, &
                                               this%packName, &
                                               this%name_model, &
                                               this%packName, &
                                               maxlist, .false., .false., &
                                               naux)
    end if
    !
    ! -- 
    naux = this%naux
    if (naux > 0) then
      !
      ! -- 
      text = '       AUXILIARY'
      idx = idx + 1
      maxlist = this%maxbound
      call this%budobj%budterm(idx)%initialize(text, &
                                               this%name_model, &
                                               this%packName, &
                                               this%name_model, &
                                               this%packName, &
                                               maxlist, .false., .false., &
                                               naux, this%auxname)
    end if
    !
    ! -- if sfr flow for each reach are written to the listing file
    if (this%iprflow /= 0) then
      call this%budobj%flowtable_df(this%iout, cellids='GWF')
    end if
    !
    ! -- return
    return
  end subroutine sfr_setup_budobj

  subroutine sfr_fill_budobj(this)
! ******************************************************************************
! sfr_fill_budobj -- copy flow terms into this%budobj
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(SfrType) :: this
    ! -- local
    integer(I4B) :: naux
    integer(I4B) :: i, n, n1, n2
    integer(I4B) :: ii
    integer(I4B) :: idx
    integer(I4B) :: idiv
    integer(I4B) :: jpos
    real(DP) :: q
    real(DP) :: qt
    real(DP) :: d
    real(DP) :: a
    ! -- formats
! -----------------------------------------------------------------------------
    !
    ! -- initialize counter
    idx = 0
    !
    ! -- FLOW JA FACE
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%nconn)
    do n = 1, this%maxbound
      n1 = n
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        n2 = this%ja(i)
        ! flow to downstream reaches
        if (this%idir(i) < 0) then
          qt = this%dsflow(n)
          q = -this%qconn(i)
        ! flow from upstream reaches
        else
          qt = this%usflow(n)
          do ii = this%ia(n2) + 1, this%ia(n2+1) - 1
            if (this%idir(ii) > 0) cycle
            if (this%ja(ii) /= n) cycle
            q = this%qconn(ii)
            exit
          end do
        end if
        ! calculate flow area
        call this%sfr_rectch_depth(n, qt, d)
        this%qauxcbc(1) = d * this%width(n)
        call this%budobj%budterm(idx)%update_term(n1, n2, q, this%qauxcbc)
      end do
    end do
    !
    ! -- GWF (LEAKAGE)
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound - this%ianynone)
    do n = 1, this%maxbound
      n2 = this%igwfnode(n)
      if (n2 > 0) then
        this%qauxcbc(1) = this%width(n) * this%length(n)
        q = -this%gwflow(n)
        call this%budobj%budterm(idx)%update_term(n, n2, q, this%qauxcbc)
      end if
    end do
    !
    ! -- RAIN
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound)
    do n = 1, this%maxbound
      a = this%surface_area(n)
      q = this%rain(n) * a
      call this%budobj%budterm(idx)%update_term(n, n, q)
    end do
    !
    ! -- EVAPORATION
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound)
    do n = 1, this%maxbound
      q = -this%simevap(n)
      call this%budobj%budterm(idx)%update_term(n, n, q)
    end do
    !
    ! -- RUNOFF
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound)
    do n = 1, this%maxbound
      q = this%simrunoff(n)
      call this%budobj%budterm(idx)%update_term(n, n, q)
    end do
    !
    ! -- INFLOW
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound)
    do n = 1, this%maxbound
      q = this%inflow(n)
      call this%budobj%budterm(idx)%update_term(n, n, q)
    end do
    !
    ! -- EXTERNAL OUTFLOW
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound)
    do n = 1, this%maxbound
      q = DZERO
      do i = this%ia(n) + 1, this%ia(n+1) - 1
        if (this%idir(i) > 0) cycle
        idiv = this%idiv(i)
        if (idiv > 0) then
          jpos = this%iadiv(n) + idiv - 1
          q = q + this%divq(jpos)
        else
          q = q + this%qconn(i)
        end if
      end do
      q = q - this%dsflow(n)
      if (this%imover == 1) then
        q = q + this%pakmvrobj%get_qtomvr(n)
      end if
      call this%budobj%budterm(idx)%update_term(n, n, q)
    end do
    !
    ! -- STORAGE
    idx = idx + 1
    call this%budobj%budterm(idx)%reset(this%maxbound)
    do n = 1, this%maxbound
      q = DZERO
      d = this%depth(n)
      a = this%width(n) * this%length(n)
      this%qauxcbc(1) = a * d
      call this%budobj%budterm(idx)%update_term(n, n, q, this%qauxcbc)
    end do
    !
    ! -- MOVER
    if (this%imover == 1) then
      ! 
      ! -- FROM MOVER
      idx = idx + 1
      call this%budobj%budterm(idx)%reset(this%maxbound)
      do n = 1, this%maxbound
        q = this%pakmvrobj%get_qfrommvr(n)
        call this%budobj%budterm(idx)%update_term(n, n, q)
      end do
      !
      ! -- TO MOVER
      idx = idx + 1
      call this%budobj%budterm(idx)%reset(this%maxbound)
      do n = 1, this%maxbound
        q = this%pakmvrobj%get_qtomvr(n)
        if (q > DZERO) then
          q = -q
        end if
        call this%budobj%budterm(idx)%update_term(n, n, q)
      end do
    end if
    !
    ! -- AUXILIARY VARIABLES
    naux = this%naux
    if (naux > 0) then
      idx = idx + 1
      call this%budobj%budterm(idx)%reset(this%maxbound)
      do n = 1, this%maxbound
        q = DZERO
        call this%budobj%budterm(idx)%update_term(n, n, q, this%auxvar(:, n))
      end do
    end if
    !
    ! --Terms are filled, now accumulate them for this time step
    call this%budobj%accumulate_terms()
    !
    ! -- return
    return
  end subroutine sfr_fill_budobj

  subroutine sfr_setup_tableobj(this)
! ******************************************************************************
! sfr_setup_tableobj -- Set up the table object that is used to write the sfr 
!                       stage data. The terms listed here must correspond in  
!                       number and order to the ones written to the stage table 
!                       in the sfr_ot method.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH, LENBUDTXT
    ! -- dummy
    class(SfrType) :: this
    ! -- local
    integer(I4B) :: nterms
    character(len=LINELENGTH) :: title
    character(len=LINELENGTH) :: text
! ------------------------------------------------------------------------------
    !
    ! -- setup stage table
    if (this%iprhed > 0) then
      !
      ! -- Determine the number of sfr budget terms. These are fixed for 
      !    the simulation and cannot change.  This includes FLOW-JA-FACE
      !    so they can be written to the binary budget files, but these internal
      !    flows are not included as part of the budget table.
      nterms = 8
      if (this%inamedbound == 1) then
        nterms = nterms + 1
      end if
      !
      ! -- set up table title
      title = trim(adjustl(this%text)) // ' PACKAGE (' //                        &
              trim(adjustl(this%packName)) //') STAGES FOR EACH CONTROL VOLUME'
      !
      ! -- set up stage tableobj
      call table_cr(this%stagetab, this%packName, title)
      call this%stagetab%table_df(this%maxbound, nterms, this%iout,              &
                                  transient=.TRUE.)
      !
      ! -- Go through and set up table budget term
      if (this%inamedbound == 1) then
        text = 'NAME'
        call this%stagetab%initialize_column(text, 20, alignment=TABLEFT)
      end if
      !
      ! -- reach number
      text = 'NUMBER'
      call this%stagetab%initialize_column(text, 10, alignment=TABCENTER)
      !
      ! -- cellids
      text = 'CELLID'
      call this%stagetab%initialize_column(text, 20, alignment=TABLEFT)
      !
      ! -- reach stage
      text = 'STAGE'
      call this%stagetab%initialize_column(text, 12, alignment=TABCENTER)
      !
      ! -- reach depth
      text = 'DEPTH'
      call this%stagetab%initialize_column(text, 12, alignment=TABCENTER)
      !
      ! -- reach width
      text = 'WIDTH'
      call this%stagetab%initialize_column(text, 12, alignment=TABCENTER)
      !
      ! -- gwf head
      text = 'GWF HEAD'
      call this%stagetab%initialize_column(text, 12, alignment=TABCENTER)
      !
      ! -- streambed conductance
      text = 'STREAMBED CONDUCTANCE'
      call this%stagetab%initialize_column(text, 12, alignment=TABCENTER)
      !
      ! -- streambed gradient
      text = 'STREAMBED GRADIENT'
      call this%stagetab%initialize_column(text, 12, alignment=TABCENTER)
    end if
    !
    ! -- return
    return
  end subroutine sfr_setup_tableobj
  
  

  ! -- geometry functions
  function area_wet(this, n, depth)
! ******************************************************************************
! area_wet -- return wetted area
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- return
    real(DP) :: area_wet
    ! -- dummy
    class(SfrType) :: this
    integer(I4B), intent(in) :: n
    real(DP), intent(in) :: depth
! ------------------------------------------------------------------------------
    !
    ! -- Calculate area
    area_wet = depth * this%width(n)
    !
    ! -- Return
    return
  end function area_wet
  
  
  function perimeter_wet(this, n)
! ******************************************************************************
! perimeter_wet -- return wetted perimeter
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- return
    real(DP) :: perimeter_wet
    ! -- dummy
    class(SfrType) :: this
    integer(I4B), intent(in) :: n
! ------------------------------------------------------------------------------
    !
    ! -- Calculate wetted perimeter
    perimeter_wet = this%width(n)
    !
    ! -- return
    return
  end function perimeter_wet

  function surface_area(this, n)
! ******************************************************************************
! surface_area -- return surface area
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- return variable
    real(DP) :: surface_area
    ! -- dummy
    class(SfrType) :: this
    integer(I4B), intent(in) :: n
! ------------------------------------------------------------------------------
    !
    ! -- Calculate surface area
    surface_area = this%width(n) * this%length(n)
    !
    ! -- Return
    return
  end function surface_area  
  
  function surface_area_wet(this, n, depth)
! ******************************************************************************
! area_wet -- return wetted surface area
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- return
    real(DP) :: surface_area_wet
    ! -- dummy
    class(SfrType) :: this
    integer(I4B), intent(in) :: n
    real(DP), intent(in) :: depth
    ! -- local
    real(DP) :: top_width
! ------------------------------------------------------------------------------
    !
    ! -- Calculate surface area
    top_width = this%top_width_wet(n, depth)
    surface_area_wet = top_width * this%length(n)
    !
    ! -- Return
    return
  end function surface_area_wet
  
  function top_width_wet(this, n, depth)
! ******************************************************************************
! area_wet -- return wetted surface area
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DEM5, DZERO
    ! -- return
    real(DP) :: top_width_wet
    ! -- dummy
    class(SfrType) :: this
    integer(I4B), intent(in) :: n
    real(DP), intent(in) :: depth
    ! -- local
    real(DP) :: sat
! ------------------------------------------------------------------------------
    !
    ! -- Calculate surface area
    sat = sCubicSaturation(DEM5, DZERO, depth, DEM5)
    top_width_wet = this%width(n) * sat
    !
    ! -- Return
    return
  end function top_width_wet

  subroutine sfr_activate_density(this)
! ******************************************************************************
! sfr_activate_density -- Activate addition of density terms
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- module
    use MemoryManagerModule, only: mem_reallocate
    ! -- dummy
    class(SfrType),intent(inout) :: this
    ! -- local
    integer(I4B) :: i, j
    ! -- formats
! ------------------------------------------------------------------------------
    !
    ! -- Set idense and reallocate denseterms to be of size MAXBOUND
    this%idense = 1
    call mem_reallocate(this%denseterms, 3, this%MAXBOUND, 'DENSETERMS', &
                        this%memoryPath)
    do i = 1, this%maxbound
      do j = 1, 3
        this%denseterms(j, i) = DZERO
      end do
    end do
    write(this%iout,'(/1x,a)') 'DENSITY TERMS HAVE BEEN ACTIVATED FOR SFR &
      &PACKAGE: ' // trim(adjustl(this%packName))
    !
    ! -- return
    return
  end subroutine sfr_activate_density

  subroutine sfr_calculate_density_exchange(this, n, stage, head, cond,        &
                                            bots, flow, gwfhcof, gwfrhs)
! ******************************************************************************
! sfr_calculate_density_exchange -- Calculate the groundwater-sfr density 
!                                   exchange terms.
!
! -- Arguments are as follows:
!     n           : sfr reach number
!     stage       : sfr stage
!     head        : gwf head
!     cond        : conductance
!     bots        : bottom elevation of the stream
!     flow        : calculated flow, updated here with density terms
!     gwfhcof     : gwf head coefficient, updated here with density terms
!     gwfrhs      : gwf right-hand-side value, updated here with density terms
!
! -- Member variable used here
!     denseterms  : shape (3, MAXBOUND), filled by buoyancy package
!                     col 1 is relative density of sfr (densesfr / denseref)
!                     col 2 is relative density of gwf cell (densegwf / denseref)
!                     col 3 is elevation of gwf cell
!
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(SfrType),intent(inout) :: this
    integer(I4B), intent(in) :: n
    real(DP), intent(in) :: stage
    real(DP), intent(in) :: head
    real(DP), intent(in) :: cond
    real(DP), intent(in) :: bots
    real(DP), intent(inout) :: flow
    real(DP), intent(inout) :: gwfhcof
    real(DP), intent(inout) :: gwfrhs
    ! -- local
    real(DP) :: ss
    real(DP) :: hh
    real(DP) :: havg
    real(DP) :: rdensesfr
    real(DP) :: rdensegwf
    real(DP) :: rdenseavg
    real(DP) :: elevsfr
    real(DP) :: elevgwf
    real(DP) :: elevavg
    real(DP) :: d1
    real(DP) :: d2
    logical :: stage_below_bot
    logical :: head_below_bot
    ! -- formats
! ------------------------------------------------------------------------------
    !
    ! -- Set sfr density to sfr density or gwf density
    if (stage >= bots) then
      ss = stage
      stage_below_bot = .false.
      rdensesfr = this%denseterms(1, n)  ! sfr rel density
    else
      ss = bots
      stage_below_bot = .true.
      rdensesfr = this%denseterms(2, n)  ! gwf rel density
    end if
    !
    ! -- set hh to head or bots
    if (head >= bots) then
      hh = head
      head_below_bot = .false.
      rdensegwf = this%denseterms(2, n)  ! gwf rel density
    else
      hh = bots
      head_below_bot = .true.
      rdensegwf = this%denseterms(1, n)  ! sfr rel density
    end if
    !
    ! -- todo: hack because denseterms not updated in a cf calculation
    if (rdensegwf == DZERO) return
    !
    ! -- Update flow
    if (stage_below_bot .and. head_below_bot) then
      !
      ! -- flow is zero, so no terms are updated
      !
    else
      !
      ! -- calulate average relative density
      rdenseavg = DHALF * (rdensesfr + rdensegwf)
      !
      ! -- Add contribution of first density term: 
      !      cond * (denseavg/denseref - 1) * (hgwf - hsfr)
      d1 = cond * (rdenseavg - DONE) 
      gwfhcof = gwfhcof - d1
      gwfrhs = gwfrhs - d1 * ss
      d1 = d1 * (hh - ss)
      flow = flow + d1
      !
      ! -- Add second density term if stage and head not below bottom
      if (.not. stage_below_bot .and. .not. head_below_bot) then
        !
        ! -- Add contribution of second density term:
        !      cond * (havg - elevavg) * (densegwf - densesfr) / denseref
        elevgwf = this%denseterms(3, n)
        elevsfr = bots
        elevavg = DHALF * (elevsfr + elevgwf)
        havg = DHALF * (hh + ss)
        d2 = cond * (havg - elevavg) * (rdensegwf - rdensesfr)
        gwfrhs = gwfrhs + d2
        flow = flow + d2
      end if
    end if
    !
    ! -- return
    return
  end subroutine sfr_calculate_density_exchange


end module SfrModule
