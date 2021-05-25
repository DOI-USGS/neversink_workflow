! -- Unsaturated Zone Flow Transport Module
! -- todo: what to do about reactions in uzf?  Decay?
! -- todo: save the uzt concentration into the uzt aux variable?
! -- todo: calculate the uzf DENSE aux variable using concentration?
! -- todo: GWD and GWD-TO-MVR do not seem to be included; prob in UZF?
!
! UZF flows (flowbudptr)     index var    UZT term              Transport Type
!---------------------------------------------------------------------------------

! -- terms from UZF that will be handled by parent APT Package
! FLOW-JA-FACE              idxbudfjf     FLOW-JA-FACE          cv2cv
! GWF (aux FLOW-AREA)       idxbudgwf     GWF                   uzf2gwf
! STORAGE (aux VOLUME)      idxbudsto     none                  used for water volumes
! FROM-MVR                  idxbudfmvr    FROM-MVR              q * cext = this%qfrommvr(:)
! AUXILIARY                 none          none                  none
! none                      none          STORAGE (aux MASS)    
! none                      none          AUXILIARY             none

! -- terms from UZF that need to be handled here
! INFILTRATION              idxbudinfl    INFILTRATION          q < 0: q * cwell, else q * cuser
! REJ-INF                   idxbudrinf    REJ-INF               q * cuzt
! UZET                      idxbuduzet    UZET                  q * cet
! REJ-INF-TO-MVR            idxbudritm    REJ-INF-TO-MVR        q * cinfil?

! -- terms from UZF that should be skipped
  
  
module GwtUztModule

  use KindModule, only: DP, I4B
  use ConstantsModule, only: DZERO, DONE, LINELENGTH
  use SimModule, only: store_error, ustop
  use BndModule, only: BndType, GetBndFromList
  use GwtFmiModule, only: GwtFmiType
  use UzfModule, only: UzfType
  use GwtAptModule, only: GwtAptType
  
  implicit none
  
  public uzt_create
  
  character(len=*), parameter :: ftype = 'UZT'
  character(len=*), parameter :: flowtype = 'UZF'
  character(len=16)           :: text  = '             UZT'
  
  type, extends(GwtAptType) :: GwtUztType
    
    integer(I4B), pointer                                  :: idxbudinfl => null()  ! index of uzf infiltration terms in flowbudptr
    integer(I4B), pointer                                  :: idxbudrinf => null()  ! index of rejected infiltration terms in flowbudptr
    integer(I4B), pointer                                  :: idxbuduzet => null()  ! index of unsat et terms in flowbudptr
    integer(I4B), pointer                                  :: idxbudritm => null()  ! index of rej infil to mover rate to mover terms in flowbudptr
    real(DP), dimension(:), pointer, contiguous            :: concinfl => null()    ! infiltration concentration
    real(DP), dimension(:), pointer, contiguous            :: concuzet => null()    ! unsat et concentration

  contains
  
    procedure :: bnd_da => uzt_da
    procedure :: allocate_scalars
    procedure :: apt_allocate_arrays => uzt_allocate_arrays
    procedure :: find_apt_package => find_uzt_package
    procedure :: pak_fc_expanded => uzt_fc_expanded
    procedure :: pak_solve => uzt_solve
    procedure :: pak_get_nbudterms => uzt_get_nbudterms
    procedure :: pak_setup_budobj => uzt_setup_budobj
    procedure :: pak_fill_budobj => uzt_fill_budobj
    procedure :: uzt_infl_term
    procedure :: uzt_rinf_term
    procedure :: uzt_uzet_term
    procedure :: uzt_ritm_term
    procedure :: pak_df_obs => uzt_df_obs
    procedure :: pak_bd_obs => uzt_bd_obs
    procedure :: pak_set_stressperiod => uzt_set_stressperiod
    
  end type GwtUztType

  contains  
  
  subroutine uzt_create(packobj, id, ibcnum, inunit, iout, namemodel, pakname, &
                        fmi)
! ******************************************************************************
! uzt_create -- Create a New UZT Package
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(BndType), pointer :: packobj
    integer(I4B),intent(in) :: id
    integer(I4B),intent(in) :: ibcnum
    integer(I4B),intent(in) :: inunit
    integer(I4B),intent(in) :: iout
    character(len=*), intent(in) :: namemodel
    character(len=*), intent(in) :: pakname
    type(GwtFmiType), pointer :: fmi
    ! -- local
    type(GwtUztType), pointer :: uztobj
! ------------------------------------------------------------------------------
    !
    ! -- allocate the object and assign values to object variables
    allocate(uztobj)
    packobj => uztobj
    !
    ! -- create name and memory path
    call packobj%set_names(ibcnum, namemodel, pakname, ftype)
    packobj%text = text
    !
    ! -- allocate scalars
    call uztobj%allocate_scalars()
    !
    ! -- initialize package
    call packobj%pack_initialize()

    packobj%inunit = inunit
    packobj%iout = iout
    packobj%id = id
    packobj%ibcnum = ibcnum
    packobj%ncolbnd = 1
    packobj%iscloc = 1
    
    ! -- Store pointer to flow model interface.  When the GwfGwt exchange is
    !    created, it sets fmi%bndlist so that the GWT model has access to all
    !    the flow packages
    uztobj%fmi => fmi
    !
    ! -- return
    return
  end subroutine uzt_create

  subroutine find_uzt_package(this)
! ******************************************************************************
! find corresponding uzt package
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_allocate
    ! -- dummy
    class(GwtUztType) :: this
    ! -- local
    character(len=LINELENGTH) :: errmsg
    class(BndType), pointer :: packobj
    integer(I4B) :: ip, icount
    integer(I4B) :: nbudterm
    logical :: found
! ------------------------------------------------------------------------------
    !
    ! -- Initialize found to false, and error later if flow package cannot
    !    be found
    found = .false.
    !
    ! -- If user is specifying flows in a binary budget file, then set up
    !    the budget file reader, otherwise set a pointer to the flow package
    !    budobj
    if (this%fmi%flows_from_file) then
      call this%fmi%set_aptbudobj_pointer(this%flowpackagename, this%flowbudptr)
      if (associated(this%flowbudptr)) found = .true.
      !
    else
      if (associated(this%fmi%gwfbndlist)) then
        ! -- Look through gwfbndlist for a flow package with the same name as 
        !    this transport package name
        do ip = 1, this%fmi%gwfbndlist%Count()
          packobj => GetBndFromList(this%fmi%gwfbndlist, ip)
          if (packobj%packName == this%flowpackagename) then
            found = .true.
            !
            ! -- store BndType pointer to packobj, and then
            !    use the select type to point to the budobj in flow package
            this%flowpackagebnd => packobj
            select type (packobj)
              type is (UzfType)
                this%flowbudptr => packobj%budobj
            end select
          end if
          if (found) exit
        end do
      end if
    end if
    !
    ! -- error if flow package not found
    if (.not. found) then
      write(errmsg, '(a)') 'COULD NOT FIND FLOW PACKAGE WITH NAME '&
                            &// trim(adjustl(this%flowpackagename)) // '.'
      call store_error(errmsg)
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- allocate space for idxbudssm, which indicates whether this is a 
    !    special budget term or one that is a general source and sink
    nbudterm = this%flowbudptr%nbudterm
    call mem_allocate(this%idxbudssm, nbudterm, 'IDXBUDSSM', this%memoryPath)
    !
    ! -- Process budget terms and identify special budget terms
    write(this%iout, '(/, a, a)') &
      'PROCESSING ' // ftype // ' INFORMATION FOR ', this%packName
    write(this%iout, '(a)') '  IDENTIFYING FLOW TERMS IN ' // flowtype // ' PACKAGE'
    write(this%iout, '(a, i0)') &
      '  NUMBER OF ' // flowtype // ' = ', this%flowbudptr%ncv
    icount = 1
    do ip = 1, this%flowbudptr%nbudterm
      select case(trim(adjustl(this%flowbudptr%budterm(ip)%flowtype)))
      case('FLOW-JA-FACE')
        this%idxbudfjf = ip
        this%idxbudssm(ip) = 0
      case('GWF')
        this%idxbudgwf = ip
        this%idxbudssm(ip) = 0
      case('STORAGE')
        this%idxbudsto = ip
        this%idxbudssm(ip) = 0
      case('INFILTRATION')
        this%idxbudinfl = ip
        this%idxbudssm(ip) = 0
      case('REJ-INF')
        this%idxbudrinf = ip
        this%idxbudssm(ip) = 0
      case('UZET')
        this%idxbuduzet = ip
        this%idxbudssm(ip) = 0
      case('REJ-INF-TO-MVR')
        this%idxbudritm = ip
        this%idxbudssm(ip) = 0
      case('TO-MVR')
        this%idxbudtmvr = ip
        this%idxbudssm(ip) = 0
      case('FROM-MVR')
        this%idxbudfmvr = ip
        this%idxbudssm(ip) = 0
      case('AUXILIARY')
        this%idxbudaux = ip
        this%idxbudssm(ip) = 0
      case default
        !
        ! -- set idxbudssm equal to a column index for where the concentrations
        !    are stored in the concbud(nbudssm, ncv) array
        this%idxbudssm(ip) = icount
        icount = icount + 1
      end select
      write(this%iout, '(a, i0, " = ", a,/, a, i0)') &
        '  TERM ', ip, trim(adjustl(this%flowbudptr%budterm(ip)%flowtype)), &
        '   MAX NO. OF ENTRIES = ', this%flowbudptr%budterm(ip)%maxlist
    end do
    write(this%iout, '(a, //)') 'DONE PROCESSING ' // ftype // ' INFORMATION'
    !
    ! -- Return
    return
  end subroutine find_uzt_package

  subroutine uzt_fc_expanded(this, rhs, ia, idxglo, amatsln)
! ******************************************************************************
! uzt_fc_expanded -- this will be called from GwtAptType%apt_fc_expanded()
!   in order to add matrix terms specifically for this package
! ****************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwtUztType) :: this
    real(DP), dimension(:), intent(inout) :: rhs
    integer(I4B), dimension(:), intent(in) :: ia
    integer(I4B), dimension(:), intent(in) :: idxglo
    real(DP), dimension(:), intent(inout) :: amatsln
    ! -- local
    integer(I4B) :: j, n1, n2
    integer(I4B) :: iloc
    integer(I4B) :: iposd
    real(DP) :: rrate
    real(DP) :: rhsval
    real(DP) :: hcofval
! ------------------------------------------------------------------------------
    !
    ! -- add infiltration contribution
    if (this%idxbudinfl /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbudinfl)%nlist
        call this%uzt_infl_term(j, n1, n2, rrate, rhsval, hcofval)
        iloc = this%idxlocnode(n1)
        iposd = this%idxpakdiag(n1)
        amatsln(iposd) = amatsln(iposd) + hcofval
        rhs(iloc) = rhs(iloc) + rhsval
      end do
    end if
    !
    ! -- add rejected infiltration contribution
    if (this%idxbudrinf /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbudrinf)%nlist
        call this%uzt_rinf_term(j, n1, n2, rrate, rhsval, hcofval)
        iloc = this%idxlocnode(n1)
        iposd = this%idxpakdiag(n1)
        amatsln(iposd) = amatsln(iposd) + hcofval
        rhs(iloc) = rhs(iloc) + rhsval
      end do
    end if
    !
    ! -- add unsaturated et contribution
    if (this%idxbuduzet /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbuduzet)%nlist
        call this%uzt_uzet_term(j, n1, n2, rrate, rhsval, hcofval)
        iloc = this%idxlocnode(n1)
        iposd = this%idxpakdiag(n1)
        amatsln(iposd) = amatsln(iposd) + hcofval
        rhs(iloc) = rhs(iloc) + rhsval
      end do
    end if
    !
    ! -- add rejected infiltration to mover contribution
    if (this%idxbudritm /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbudritm)%nlist
        call this%uzt_ritm_term(j, n1, n2, rrate, rhsval, hcofval)
        iloc = this%idxlocnode(n1)
        iposd = this%idxpakdiag(n1)
        amatsln(iposd) = amatsln(iposd) + hcofval
        rhs(iloc) = rhs(iloc) + rhsval
      end do
    end if
    !
    ! -- Return
    return
  end subroutine uzt_fc_expanded

  subroutine uzt_solve(this)
! ******************************************************************************
! uzt_solve -- add terms specific to the unsaturated zone to the explicit 
!              unsaturated-zone solve
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwtUztType) :: this
    ! -- local
    integer(I4B) :: j
    integer(I4B) :: n1, n2
    real(DP) :: rrate
! ------------------------------------------------------------------------------
    !
    ! -- add infiltration contribution
    if (this%idxbudinfl /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbudinfl)%nlist
        call this%uzt_infl_term(j, n1, n2, rrate)
        this%dbuff(n1) = this%dbuff(n1) + rrate
      end do
    end if
    !
    ! -- add rejected infiltration contribution
    if (this%idxbudrinf /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbudrinf)%nlist
        call this%uzt_rinf_term(j, n1, n2, rrate)
        this%dbuff(n1) = this%dbuff(n1) + rrate
      end do
    end if
    !
    ! -- add unsaturated et contribution
    if (this%idxbuduzet /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbuduzet)%nlist
        call this%uzt_uzet_term(j, n1, n2, rrate)
        this%dbuff(n1) = this%dbuff(n1) + rrate
      end do
    end if
    !
    ! -- add rejected infiltration to mover contribution
    if (this%idxbudritm /= 0) then
      do j = 1, this%flowbudptr%budterm(this%idxbudritm)%nlist
        call this%uzt_ritm_term(j, n1, n2, rrate)
        this%dbuff(n1) = this%dbuff(n1) + rrate
      end do
    end if
    !
    ! -- Return
    return
  end subroutine uzt_solve
  
  function uzt_get_nbudterms(this) result(nbudterms)
! ******************************************************************************
! uzt_get_nbudterms -- function to return the number of budget terms just for
!   this package.  This overrides function in parent.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwtUztType) :: this
    ! -- return
    integer(I4B) :: nbudterms
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- Number of budget terms is 4
    nbudterms = 0
    if (this%idxbudinfl /= 0) nbudterms = nbudterms + 1
    if (this%idxbudrinf /= 0) nbudterms = nbudterms + 1
    if (this%idxbuduzet /= 0) nbudterms = nbudterms + 1
    if (this%idxbudritm /= 0) nbudterms = nbudterms + 1
    !
    ! -- Return
    return
  end function uzt_get_nbudterms
  
  subroutine uzt_setup_budobj(this, idx)
! ******************************************************************************
! uzt_setup_budobj -- Set up the budget object that stores all the unsaturated-
!                     zone flows
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LENBUDTXT
    ! -- dummy
    class(GwtUztType) :: this
    integer(I4B), intent(inout) :: idx
    ! -- local
    integer(I4B) :: maxlist, naux
    character(len=LENBUDTXT) :: text
! ------------------------------------------------------------------------------
    !
    ! -- 
    text = '    INFILTRATION'
    idx = idx + 1
    maxlist = this%flowbudptr%budterm(this%idxbudinfl)%maxlist
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
    if (this%idxbudrinf /= 0) then
      text = '         REJ-INF'
      idx = idx + 1
      maxlist = this%flowbudptr%budterm(this%idxbudrinf)%maxlist
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
    if (this%idxbuduzet /= 0) then
      text = '            UZET'
      idx = idx + 1
      maxlist = this%flowbudptr%budterm(this%idxbuduzet)%maxlist
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
    if (this%idxbudritm /= 0) then
      text = '  INF-REJ-TO-MVR'
      idx = idx + 1
      maxlist = this%flowbudptr%budterm(this%idxbudritm)%maxlist
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
    ! -- return
    return
  end subroutine uzt_setup_budobj

  subroutine uzt_fill_budobj(this, idx, x, ccratin, ccratout)
! ******************************************************************************
! uzt_fill_budobj -- copy flow terms into this%budobj
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwtUztType) :: this
    integer(I4B), intent(inout) :: idx
    real(DP), dimension(:), intent(in) :: x
    real(DP), intent(inout) :: ccratin
    real(DP), intent(inout) :: ccratout
    ! -- local
    integer(I4B) :: j, n1, n2
    integer(I4B) :: nlist
    real(DP) :: q
    ! -- formats
! -----------------------------------------------------------------------------
    
    ! -- INFILTRATION
    idx = idx + 1
    nlist = this%flowbudptr%budterm(this%idxbudinfl)%nlist
    call this%budobj%budterm(idx)%reset(nlist)
    do j = 1, nlist
      call this%uzt_infl_term(j, n1, n2, q)
      call this%budobj%budterm(idx)%update_term(n1, n2, q)
      call this%apt_accumulate_ccterm(n1, q, ccratin, ccratout)
    end do
    
    ! -- REJ-INF
    if (this%idxbudrinf /= 0) then
      idx = idx + 1
      nlist = this%flowbudptr%budterm(this%idxbudrinf)%nlist
      call this%budobj%budterm(idx)%reset(nlist)
      do j = 1, nlist
        call this%uzt_rinf_term(j, n1, n2, q)
        call this%budobj%budterm(idx)%update_term(n1, n2, q)
        call this%apt_accumulate_ccterm(n1, q, ccratin, ccratout)
      end do
    end if
    
    ! -- UZET
    if (this%idxbuduzet /= 0) then
      idx = idx + 1
      nlist = this%flowbudptr%budterm(this%idxbuduzet)%nlist
      call this%budobj%budterm(idx)%reset(nlist)
      do j = 1, nlist
        call this%uzt_uzet_term(j, n1, n2, q)
        call this%budobj%budterm(idx)%update_term(n1, n2, q)
        call this%apt_accumulate_ccterm(n1, q, ccratin, ccratout)
      end do
    end if
    
    ! -- REJ-INF-TO-MVR
    if (this%idxbudritm /= 0) then
      idx = idx + 1
      nlist = this%flowbudptr%budterm(this%idxbudritm)%nlist
      call this%budobj%budterm(idx)%reset(nlist)
      do j = 1, nlist
        call this%uzt_ritm_term(j, n1, n2, q)
        call this%budobj%budterm(idx)%update_term(n1, n2, q)
        call this%apt_accumulate_ccterm(n1, q, ccratin, ccratout)
      end do
    end if
    
    !
    ! -- return
    return
  end subroutine uzt_fill_budobj

  subroutine allocate_scalars(this)
! ******************************************************************************
! allocate_scalars
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_allocate
    ! -- dummy
    class(GwtUztType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- allocate scalars in GwtAptType
    call this%GwtAptType%allocate_scalars()
    !
    ! -- Allocate
    call mem_allocate(this%idxbudinfl, 'IDXBUDINFL', this%memoryPath)
    call mem_allocate(this%idxbudrinf, 'IDXBUDRINF', this%memoryPath)
    call mem_allocate(this%idxbuduzet, 'IDXBUDUZET', this%memoryPath)
    call mem_allocate(this%idxbudritm, 'IDXBUDRITM', this%memoryPath)
    ! 
    ! -- Initialize
    this%idxbudinfl = 0
    this%idxbudrinf = 0
    this%idxbuduzet = 0
    this%idxbudritm = 0
    !
    ! -- Return
    return
  end subroutine allocate_scalars

  subroutine uzt_allocate_arrays(this)
! ******************************************************************************
! uzt_allocate_arrays
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_allocate
    ! -- dummy
    class(GwtUztType), intent(inout) :: this
    ! -- local
    integer(I4B) :: n
! ------------------------------------------------------------------------------
    !    
    ! -- time series
    call mem_allocate(this%concinfl, this%ncv, 'CONCINFL', this%memoryPath)
    call mem_allocate(this%concuzet, this%ncv, 'CONCUZET', this%memoryPath)
    !
    ! -- call standard GwtApttype allocate arrays
    call this%GwtAptType%apt_allocate_arrays()
    !
    ! -- Initialize
    do n = 1, this%ncv
      this%concinfl(n) = DZERO
      this%concuzet(n) = DZERO
    end do
    !
    !
    ! -- Return
    return
  end subroutine uzt_allocate_arrays
  
  subroutine uzt_da(this)
! ******************************************************************************
! uzt_da
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_deallocate
    ! -- dummy
    class(GwtUztType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- deallocate scalars
    call mem_deallocate(this%idxbudinfl)
    call mem_deallocate(this%idxbudrinf)
    call mem_deallocate(this%idxbuduzet)
    call mem_deallocate(this%idxbudritm)
    !
    ! -- deallocate time series
    call mem_deallocate(this%concinfl)
    call mem_deallocate(this%concuzet)
    !
    ! -- deallocate scalars in GwtAptType
    call this%GwtAptType%bnd_da()
    !
    ! -- Return
    return
  end subroutine uzt_da

  subroutine uzt_infl_term(this, ientry, n1, n2, rrate, &
                           rhsval, hcofval)
! ******************************************************************************
! uzt_infl_term
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwtUztType) :: this
    integer(I4B), intent(in) :: ientry
    integer(I4B), intent(inout) :: n1
    integer(I4B), intent(inout) :: n2
    real(DP), intent(inout), optional :: rrate
    real(DP), intent(inout), optional :: rhsval
    real(DP), intent(inout), optional :: hcofval
    ! -- local
    real(DP) :: qbnd
    real(DP) :: ctmp
    real(DP) :: h, r
! ------------------------------------------------------------------------------
    n1 = this%flowbudptr%budterm(this%idxbudinfl)%id1(ientry)
    n2 = this%flowbudptr%budterm(this%idxbudinfl)%id2(ientry)
    ! -- note that qbnd is negative for negative infiltration
    qbnd = this%flowbudptr%budterm(this%idxbudinfl)%flow(ientry)
    if (qbnd < DZERO) then
      ctmp = this%xnewpak(n1)
      h = qbnd
      r = DZERO
    else
      ctmp = this%concinfl(n1)
      h = DZERO
      r = -qbnd * ctmp
    end if
    if (present(rrate)) rrate = qbnd * ctmp
    if (present(rhsval)) rhsval = r
    if (present(hcofval)) hcofval = h
    !
    ! -- return
    return
  end subroutine uzt_infl_term
  
  subroutine uzt_rinf_term(this, ientry, n1, n2, rrate, &
                           rhsval, hcofval)
! ******************************************************************************
! uzt_rinf_term
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwtUztType) :: this
    integer(I4B), intent(in) :: ientry
    integer(I4B), intent(inout) :: n1
    integer(I4B), intent(inout) :: n2
    real(DP), intent(inout), optional :: rrate
    real(DP), intent(inout), optional :: rhsval
    real(DP), intent(inout), optional :: hcofval
    ! -- local
    real(DP) :: qbnd
    real(DP) :: ctmp
! ------------------------------------------------------------------------------
    n1 = this%flowbudptr%budterm(this%idxbudrinf)%id1(ientry)
    n2 = this%flowbudptr%budterm(this%idxbudrinf)%id2(ientry)
    qbnd = this%flowbudptr%budterm(this%idxbudrinf)%flow(ientry)
    ctmp = this%concinfl(n1)
    if (present(rrate)) rrate = ctmp * qbnd
    if (present(rhsval)) rhsval = DZERO
    if (present(hcofval)) hcofval = qbnd
    !
    ! -- return
    return
  end subroutine uzt_rinf_term
  
  subroutine uzt_uzet_term(this, ientry, n1, n2, rrate, &
                           rhsval, hcofval)
! ******************************************************************************
! uzt_uzet_term
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwtUztType) :: this
    integer(I4B), intent(in) :: ientry
    integer(I4B), intent(inout) :: n1
    integer(I4B), intent(inout) :: n2
    real(DP), intent(inout), optional :: rrate
    real(DP), intent(inout), optional :: rhsval
    real(DP), intent(inout), optional :: hcofval
    ! -- local
    real(DP) :: qbnd
    real(DP) :: ctmp
    real(DP) :: omega
! ------------------------------------------------------------------------------
    n1 = this%flowbudptr%budterm(this%idxbuduzet)%id1(ientry)
    n2 = this%flowbudptr%budterm(this%idxbuduzet)%id2(ientry)
    ! -- note that qbnd is negative for uzet
    qbnd = this%flowbudptr%budterm(this%idxbuduzet)%flow(ientry)
    ctmp = this%concuzet(n1)
    if (this%xnewpak(n1) < ctmp) then
      omega = DONE
    else
      omega = DZERO
    end if
    if (present(rrate)) &
      rrate = omega * qbnd * this%xnewpak(n1) + &
              (DONE - omega) * qbnd * ctmp
    if (present(rhsval)) rhsval = - (DONE - omega) * qbnd * ctmp
    if (present(hcofval)) hcofval = omega * qbnd
    !
    ! -- return
    return
  end subroutine uzt_uzet_term
  
  subroutine uzt_ritm_term(this, ientry, n1, n2, rrate, &
                           rhsval, hcofval)
! ******************************************************************************
! uzt_ritm_term
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwtUztType) :: this
    integer(I4B), intent(in) :: ientry
    integer(I4B), intent(inout) :: n1
    integer(I4B), intent(inout) :: n2
    real(DP), intent(inout), optional :: rrate
    real(DP), intent(inout), optional :: rhsval
    real(DP), intent(inout), optional :: hcofval
    ! -- local
    real(DP) :: qbnd
    real(DP) :: ctmp
! ------------------------------------------------------------------------------
    n1 = this%flowbudptr%budterm(this%idxbudritm)%id1(ientry)
    n2 = this%flowbudptr%budterm(this%idxbudritm)%id2(ientry)
    qbnd = this%flowbudptr%budterm(this%idxbudritm)%flow(ientry)
    ctmp = this%concinfl(n1)
    if (present(rrate)) rrate = ctmp * qbnd
    if (present(rhsval)) rhsval = DZERO
    if (present(hcofval)) hcofval = qbnd
    !
    ! -- return
    return
  end subroutine uzt_ritm_term
  
  subroutine uzt_df_obs(this)
! ******************************************************************************
! uzt_df_obs -- obs are supported?
!   -- Store observation type supported by APT package.
!   -- Overrides BndType%bnd_df_obs
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use GwtAptModule, only: apt_process_obsID
    ! -- dummy
    class(GwtUztType) :: this
    ! -- local
    integer(I4B) :: indx
! ------------------------------------------------------------------------------
    !
    ! -- Store obs type and assign procedure pointer
    !    for observation type.
    call this%obs%StoreObsType('infiltration', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => apt_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for observation type.
    call this%obs%StoreObsType('rej-inf', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => apt_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for observation type.
    call this%obs%StoreObsType('uzet', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => apt_process_obsID
    !
    ! -- Store obs type and assign procedure pointer
    !    for observation type.
    call this%obs%StoreObsType('rej-inf-to-mvr', .true., indx)
    this%obs%obsData(indx)%ProcessIdPtr => apt_process_obsID
    !
    return
  end subroutine uzt_df_obs
  
  subroutine uzt_bd_obs(this, obstypeid, jj, v, found)
! ******************************************************************************
! uzt_bd_obs -- calculate observation value and pass it back to APT
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwtUztType), intent(inout) :: this
    character(len=*), intent(in) :: obstypeid
    real(DP), intent(inout) :: v
    integer(I4B), intent(in) :: jj
    logical, intent(inout) :: found
    ! -- local
    integer(I4B) :: n1, n2
! ------------------------------------------------------------------------------
    !
    found = .true.
    select case (obstypeid)
      case ('INFILTRATION')
        if (this%iboundpak(jj) /= 0 .and. this%idxbudinfl > 0) then
          call this%uzt_infl_term(jj, n1, n2, v)
        end if
      case ('REJ-INF')
        if (this%iboundpak(jj) /= 0 .and. this%idxbudrinf > 0) then
          call this%uzt_rinf_term(jj, n1, n2, v)
        end if
      case ('UZET')
        if (this%iboundpak(jj) /= 0 .and. this%idxbuduzet > 0) then
          call this%uzt_uzet_term(jj, n1, n2, v)
        end if
      case ('REJ-INF-TO-MVR')
        if (this%iboundpak(jj) /= 0 .and. this%idxbudritm > 0) then
          call this%uzt_ritm_term(jj, n1, n2, v)
        end if
      case default
        found = .false.
    end select
    !
    return
  end subroutine uzt_bd_obs

  subroutine uzt_set_stressperiod(this, itemno, keyword, found)
! ******************************************************************************
! uzt_set_stressperiod -- Set a stress period attribute for using keywords.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    use TimeSeriesManagerModule, only: read_value_or_time_series_adv
    ! -- dummy
    class(GwtUztType),intent(inout) :: this
    integer(I4B), intent(in) :: itemno
    character(len=*), intent(in) :: keyword
    logical, intent(inout) :: found
    ! -- local
    character(len=LINELENGTH) :: text
    integer(I4B) :: ierr
    integer(I4B) :: jj
    real(DP), pointer :: bndElem => null()
    ! -- formats
! ------------------------------------------------------------------------------
    !
    ! INFILTRATION <infiltration>
    ! UZET <uzet>
    !
    found = .true.
    select case (keyword)
      case ('INFILTRATION')
        ierr = this%apt_check_valid(itemno)
        if (ierr /= 0) then
          goto 999
        end if
        call this%parser%GetString(text)
        jj = 1
        bndElem => this%concinfl(itemno)
        call read_value_or_time_series_adv(text, itemno, jj, bndElem, this%packName, &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'INFILTRATION')
      case ('UZET')
        ierr = this%apt_check_valid(itemno)
        if (ierr /= 0) then
          goto 999
        end if
        call this%parser%GetString(text)
        jj = 1
        bndElem => this%concuzet(itemno)
        call read_value_or_time_series_adv(text, itemno, jj, bndElem, this%packName, &
                                           'BND', this%tsManager, this%iprpak,   &
                                           'UZET')
      case default
        !
        ! -- keyword not recognized so return to caller with found = .false.
        found = .false.
    end select
    !
999 continue      
    !
    ! -- return
    return
  end subroutine uzt_set_stressperiod


end module GwtUztModule